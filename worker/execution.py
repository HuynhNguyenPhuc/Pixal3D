"""Task Execution Logic for Worker Process."""

import json
import os
import threading
import time
import traceback
from typing import Optional

import app_state
import config
from broker.client import redis_client
from broker.state import (
    has_task_timed_out,
    record_task_heartbeat,
    remove_running_task,
    running_task_key,
    set_status_in_redis,
)
from utilities.cleanup import cleanup_task_files
from utilities.error_handling import classify_task_error
from utilities.gcloud import convert_to_gcs_url, upload_blob
from utilities.gpu import aggressive_gpu_cleanup
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def run_worker(
    uid: str,
    params: dict,
    exec_lock_key: Optional[str] = None,
    exec_lock_token: Optional[str] = None,
):
    """
    Main worker execution function for a single task.

    Args:
        uid: Unique identifier for the task.
        params: Parameters for the generation task.
        exec_lock_key: Redis key for execution lock, if using distributed locking.
        exec_lock_token: Token for verifying execution lock ownership.
    """
    # Record the start time of the task execution.
    started_at = time.time()

    # Get the worker ID.
    worker_id = app_state.worker_id or f"worker-{os.getpid()}"

    if redis_client:
        # Publish the running marker in Redis so the watchdog and status polls know this task is active.
        running_payload = {
            "uid": uid,
            "started_at": started_at,
            "last_heartbeat": started_at,
            "pid": os.getpid(),
            "worker_id": worker_id,
            "lock_token": exec_lock_token,
        }

        try:
            # Store the initial running-task payload in Redis.
            redis_client.set(
                name=running_task_key(uid),
                value=json.dumps(running_payload),
                ex=config.RUNNING_TASK_KEY_TTL_SECONDS,
            )

            # Add the heartbeat timestamp to the sorted-set index.
            record_task_heartbeat(uid, started_at)

        except Exception:
            logger.exception(f"Failed to record running task marker for uid={uid}")
            pass

    # Create a threading event to signal the heartbeat thread to stop when the task is done.
    heartbeat_stop = threading.Event()

    def heartbeat_loop():
        """Refresh the running marker and lock TTL while execution is still active."""
        while not heartbeat_stop.wait(max(config.RUNNING_HEARTBEAT_SECONDS, 1)):
            if not redis_client:
                continue

            # Rewrite the payload each cycle so the running marker always
            # reflects the latest heartbeat and current lock ownership.
            payload = {
                "uid": uid,
                "started_at": started_at,
                "last_heartbeat": time.time(),
                "pid": os.getpid(),
                "worker_id": worker_id,
                "lock_token": exec_lock_token,
            }

            try:
                # Store the updated running-task payload in Redis on each heartbeat.
                redis_client.set(
                    name=running_task_key(uid),
                    value=json.dumps(payload),
                    ex=config.RUNNING_TASK_KEY_TTL_SECONDS,
                )

                # Update the heartbeat timestamp in the sorted-set index.
                record_task_heartbeat(uid, payload["last_heartbeat"])

            except Exception:
                logger.exception(f"Failed to update heartbeat for uid={uid}")
                pass

            if exec_lock_key and exec_lock_token:
                try:
                    # Refresh the execution lock TTL only if we still own the lock.
                    if redis_client.get(exec_lock_key) == exec_lock_token:
                        redis_client.expire(exec_lock_key, config.EXEC_LOCK_TTL_SECONDS)

                except Exception:
                    logger.exception(f"Failed to refresh execution lock for uid={uid}")
                    pass

    # Start the heartbeat thread to periodically update the task heartbeat in Redis.
    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    try:
        # Aggressively clean up GPU memory before starting generation.
        aggressive_gpu_cleanup()

        # Mark the task as running in Redis before heavy work begins.
        set_status_in_redis(
            uid=uid,
            status_data={
                "status": "running",
                "hash": uid,
                "worker_id": worker_id,
            }
        )

        # If using distributed locking, verify that we still hold the lock before generation.
        if exec_lock_key and exec_lock_token and redis_client:
            try:
                current_token = redis_client.get(exec_lock_key)

                if current_token != exec_lock_token:
                    logger.warning(f"Lost execution lock for task {uid}. Skipping generation.")
                    return

            except Exception:
                logger.exception(f"Failed to verify execution lock for uid={uid}")
                pass

        # Run the actual model generation inside the worker.
        file_path = app_state.worker.generate(uid, params)

        # Do not overwrite a timeout status if watchdog already marked failure.
        if has_task_timed_out(uid):
            logger.warning(f"Task {uid} timed out during execution. Skipping completion write.")
            return

        # Validate that generation produced the expected output file.
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Output file not created: {file_path}")

        # Validate that the output file is not empty.
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"Output file is empty: {file_path}")

        logger.debug(f"✅ Output file ready: {file_size} bytes")

        # Fail fast if upload is required but the runtime has no usable GCS config.
        if not (config.GOOGLE_SERVICE_ACCOUNT and config.GCS_BUCKET):
            raise RuntimeError("GCS upload skipped - missing configuration")

        filename = os.path.basename(file_path)
        object_name = config.GCS_DESTINATION_PREFIX.format(folder_hash=uid) + filename

        # Transition to 'uploading' before touching GCS.
        # This ensures that even if the upload fails, the task status reflects that it is no longer running and has moved to a terminal state.
        set_status_in_redis(
            uid=uid,
            status_data={
                "status": "uploading",
                "hash": uid,
                "worker_id": worker_id,
                "lease_until": time.time() + config.EXEC_LOCK_TTL_SECONDS,
            }
        )

        # Upload the output file to GCS.
        try:
            upload_blob(
                bucket_name=config.GCS_BUCKET,
                source_file_name=file_path,
                destination_blob_name=object_name,
            )

            logger.info(f"✅ Upload {uid} -> {object_name}")

        except Exception as exc:
            logger.error(f"❌ Upload {uid} -> {object_name}: {exc}")
            raise

        # Write the completed status into Redis.
        set_status_in_redis(
            uid=uid,
            status_data={
                "status": "completed",
                "hash": uid,
                "filename": filename,
                "url": convert_to_gcs_url(config.GCS_BUCKET, object_name),
                "worker_id": worker_id,
            }
        )
        logger.info(f"✅ Task {uid} marked as completed in Redis.")

        # Cleanup all local files related to the task to free up disk space.
        cleanup_task_files(uid)

        # Successful completion confirms the GPU is healthy — reset the OOM counter.
        app_state.consecutive_cuda_oom_count = 0

        logger.info(f"✅ Task {uid} completed successfully in {time.time() - started_at:.2f}s")

    except TimeoutError as exc:
        logger.error(f"❌ Task {uid} timed out: {exc}")

        # Define the error response for explicit timeout failures.
        error_info = {
            "message": str(exc),
            "detail": str(exc),
            "error_code": "TIMEOUT",
            "error_type": "timeout",
            "retriable": True,
            "retry_after": 0,
            "worker_id": worker_id,
        }

        set_status_in_redis(
            uid=uid, 
            status_data={
                "status": "error", 
                "hash": uid, 
                **error_info
            }, 
            force=True
        )

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"❌ Task {uid} failed with error: {error_msg}")

        # Print the full traceback for debugging purposes.
        traceback.print_exc()

        # Classify the error to determine whether it is retriable or resource-related.
        classified = classify_task_error(exc, params)

        # Define the error response for general failures.
        error_info = {
            "message": classified.get("message", error_msg),
            "detail": classified.get("detail", error_msg),
            "error_code": classified.get("error_code", "GENERATION_FAILED"),
            "error_type": classified.get("error_type", "internal"),
            "retriable": classified.get("retriable", False),
            "retry_after": classified.get("retry_after", 0),
            "worker_id": worker_id,
        }

        set_status_in_redis(
            uid=uid, 
            status_data={
                "status": "error", 
                "hash": uid, 
                **error_info
            }, 
            force=True
        )

        # For CUDA OOM failures, do an immediate extra cleanup and track consecutive failures.  
        # A single OOM may be transient and recoverable via cleanup and retry.
        # We only flag a subprocess restart after N OOMs in a row.
        if classified.get("error_type") == "resource_exhausted":
            # Perform aggressive GPU cleanup to free up memory.
            aggressive_gpu_cleanup()

            # Increment the consecutive OOM count since this failure was classified as a resource exhaustion error.
            app_state.consecutive_cuda_oom_count += 1

            if app_state.consecutive_cuda_oom_count >= config.MAX_CONSECUTIVE_CUDA_OOM:
                app_state.needs_subprocess_restart = True
                logger.warning(
                    f"CUDA OOM #{app_state.consecutive_cuda_oom_count} for task {uid}: "
                    f"GPU state unrecoverable — subprocess restart flagged."
                )
            else:
                logger.warning(
                    f"CUDA OOM #{app_state.consecutive_cuda_oom_count} for task {uid}: cleanup performed, will retry before "
                    f"considering subprocess restart."
                )

    except BaseException as exc:
        error_msg = str(exc)
        logger.critical(f"Task {uid} failed with critical error: {error_msg}")

        # Print the full traceback for debugging purposes.
        traceback.print_exc()

        # Perform aggressive cleanup after critical failures to keep the worker healthy.
        aggressive_gpu_cleanup()

        error_info = {
            "message": "System crash",
            "detail": error_msg,
            "error_code": "SYSTEM_CRASH",
            "error_type": "internal",
            "retriable": False,
            "worker_id": worker_id,
        }

        set_status_in_redis(
            uid=uid, 
            status_data={
                "status": "error", 
                "hash": uid, 
                **error_info
            }, 
            force=True
        )

        raise

    finally:
        # Signal the heartbeat thread to stop since the task is done.
        heartbeat_stop.set()

        try:
            # Wait for the heartbeat thread to finish so it does not leak past task completion.
            heartbeat_thread.join(timeout=max(config.RUNNING_HEARTBEAT_SECONDS, 1) + 1)

        except Exception:
            pass

        # Remove the task from the running-task heartbeat tracking.
        remove_running_task(uid)

        # Aggressively clean up GPU memory after every task.
        aggressive_gpu_cleanup()

        logger.info(f"✅ Task {uid} execution finished. Cleaned up running task state.")
