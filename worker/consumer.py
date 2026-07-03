"""Task Consumer daemon."""

import json
import os
import random
import socket
import time
import uuid
from typing import Optional

import app_state
import config
from broker.client import is_redis_connection_error, redis_client, reset_redis_connection
from broker.lock import (
    LockHeartbeat, 
    acquire_exec_lock, 
    execution_lock_key, 
    release_exec_lock,
)   
from broker.queue import (
    ack_and_cleanup_stream_entry,
    ensure_consumer_group,
    invalidate_consumer_group,
    push_to_dlq,
    reconcile_queue_depth_counter,
)
from broker.state import (
    is_terminal_status,
    running_task_key,
    get_status_from_redis,
    set_status_in_redis,
)
from utilities.gpu import aggressive_gpu_cleanup
from utilities.logger import get_logger
from worker.execution import run_worker


# --- Logger --- #
logger = get_logger(__name__)

# XAUTOCLAIM support flag
xautoclaim_supported: bool = True


def _redis_retry_delay(attempt: int) -> float:
    """
    Return jittered exponential backoff delay for Redis reconnect attempts.

    Args:
        attempt (int): The current attempt number for the Redis reconnect.

    Returns:
        float: The calculated delay in seconds before the next reconnect attempt.
    """
    base_delay = min(2 ** min(attempt, 5), config.REDIS_RETRY_BACKOFF_MAX_SECONDS)
    return base_delay + random.uniform(0.1, 0.9)


def task_consumer_daemon() -> None:
    """
    Task consumer daemon that continuously polls the Redis stream for new tasks and processes them.

    Priority of processing:
    1. Tasks already assigned to THIS consumer, or pending list.
    2. Stale pending tasks left behind by dead workers (via XAUTOCLAIM).
    3. New tasks from the stream.
    """
    global xautoclaim_supported

    # Initialize Redis failure count
    redis_failure_count = 0

    # Define an unique worker session ID for this instance of the consumer.
    worker_session_id = uuid.uuid4().hex[:8]

    if not redis_client or app_state.consumer_id is None:
        logger.warning("Redis client not initialized or consumer ID not set. Task consumer cannot start.")
        return

    logger.info(f"📥 Task consumer {app_state.consumer_id} (Session: {worker_session_id}) started!")

    # Cursor for XAUTOCLAIM:
    # Redis Stream IDs have format <timestamp>-<sequence>.
    # "0-0" means start scanning from the earliest pending entry.
    reclaim_start_id = "0-0"

    def process_stream_entry(entry_id: str, data: dict, source: str = "new") -> None:
        """
        Process a single Redis stream entry representing a task.

        Args:
            entry_id (str): The ID of the stream entry.
            data (dict): The data payload of the stream entry.
            source (str): The source of the entry, either ``new`` or ``reclaimed``.
        """
        lock_key: Optional[str] = None
        lock_token: Optional[str] = None

        def _release_exec_lock():
            if lock_key and lock_token:
                release_exec_lock(lock_key, lock_token)

        uid = "<unknown>"

        try:
            # Get the UID of the task.
            uid = data.get("uid") or ""
            logger.info(f"📥 Picking up task {uid} from {source} source (entry_id={entry_id})")

            if not uid:
                logger.error(f"Stream entry {entry_id} missing 'uid'. Acknowledging and skipping.")
                ack_and_cleanup_stream_entry(entry_id)
                return

            # If Redis already marks the task as terminal, we can ack and skip immediately.
            # This provides idempotency: if we crashed after generating but before ACK, the next worker reading this from PEL will see 'completed' and skip.
            existing_status = get_status_from_redis(uid)
            if existing_status and is_terminal_status(existing_status):
                logger.info(f"Skip task {uid}: already terminal according to Redis status.")
                ack_and_cleanup_stream_entry(entry_id)
                return

            # Reclaim guards: verify that the original worker is truly stale before taking ownership.
            # This applies to both XAUTOCLAIM (reclaimed) and tasks left in our own PEL (pending) after a crash.
            if source in ["reclaimed", "pending"] and redis_client:
                try:
                    # Block reclaim if the task is in 'uploading' state and the upload lease has not yet expired. 
                    # This covers the gap where the heartbeat is gone but the worker is still pushing to GCS.
                    current_status = get_status_from_redis(uid)
                    if current_status and current_status.get("status") == "uploading":
                        lease_until = float(current_status.get("lease_until", 0) or 0)
                        if time.time() < lease_until:
                            logger.info(f"Skip {source} {uid}: upload lease still valid until {lease_until:.0f}")
                            return

                    # Get the running task detail from Redis.
                    raw = redis_client.get(running_task_key(uid))

                    if raw:
                        # Parse the running task details.
                        task_details = json.loads(raw)

                        # Get all the task information we need to determine if it is still active.
                        last_heartbeat = float(task_details.get("last_heartbeat", task_details.get("started_at", 0)) or 0)
                        marker_pid = task_details.get("pid")
                        task_lock_token = task_details.get("lock_token")
                        heartbeat_age = time.time() - last_heartbeat

                        # If the heartbeat is still fresh, skip reclaiming this entry.
                        if heartbeat_age <= config.RUNNING_HEARTBEAT_STALE_SECONDS:
                            logger.info(
                                f"Skip {source} {uid}: active heartbeat {heartbeat_age:.1f}s "
                                f"from pid={marker_pid}"
                            )
                            return

                        # If the execution lock still matches the recorded lock token from the dead worker, it means the lock is abandoned and hasn't expired yet.
                        # We must delete it to allow this consumer to acquire it.
                        current_lock_token = redis_client.get(execution_lock_key(uid))

                        if current_lock_token and task_lock_token and current_lock_token == task_lock_token:
                            logger.info(f"Reclaiming abandoned lock for {uid} from dead worker")
                            redis_client.delete(execution_lock_key(uid))

                        # If a lock still exists but the task record has no token, be conservative
                        # and skip reclaiming to avoid fighting with another worker.
                        elif current_lock_token and not task_lock_token:
                            logger.info(f"Skip {source} {uid}: stale heartbeat but unknown lock owner")
                            return

                    else:
                        # If there is no heartbeat at all, the task is definitely not running.
                        # We can safely clear any residual lock to allow this worker to acquire it.
                        if redis_client.exists(execution_lock_key(uid)):
                            logger.info(f"Reclaiming abandoned lock for {uid} (no heartbeat found)")
                            redis_client.delete(execution_lock_key(uid))

                except Exception:
                    logger.warning(f"Failed to check heartbeat for {source} task {uid}. Proceeding with processing.")

            # Acquire execution lock for the task to ensure only one worker can process it.
            lock_key = execution_lock_key(uid)
            lock_token = f"{app_state.consumer_id}:{entry_id}:{uuid.uuid4().hex}"
            lock_acquired = acquire_exec_lock(lock_key, lock_token)

            if not lock_acquired:
                logger.warning(
                    f"Skip task {uid} ({source}): failed to acquire execution lock - "
                    f"likely being processed by another worker."
                )
                return

            # Read current retry count. 
            # Count is only incremented on actual execution failure to avoid inflating on pod kills or reclaims.
            retry_key = f"task:{uid}:retries"
            max_retries = config.MAX_TASK_RETRIES

            try:
                # Get the current retry count for the task.
                raw_count = redis_client.get(retry_key)

                # If the retry count is not set, default to 0.
                retry_count = int(raw_count) if raw_count else 0
            
            except Exception:
                retry_count = 0

            # Push to DLQ if it already exceeded max retries before even starting execution to prevent wasteful GPU cycles on known-bad tasks.
            if retry_count >= max_retries:
                logger.error(f"Task {uid} already exceeded max retries ({retry_count}/{max_retries}). Moving to dead-letter queue.")

                push_to_dlq(uid, data, {
                    "failed_at": str(time.time()),
                    "retry_count": str(retry_count),
                    "consumer_id": str(app_state.consumer_id),
                    "worker_session_id": worker_session_id,
                    "hostname": socket.gethostname(),
                    "reason": "max_retries_exceeded_before_run",
                })

                # Set status to failed in Redis.
                set_status_in_redis(uid, {"status": "failed", "error": "Max retries exceeded."})
                
                # Acknowledge the stream entry and clean up to prevent further retries.
                ack_and_cleanup_stream_entry(entry_id)

                # Release the execution lock since we're marking this task as failed and won't be processing it.
                _release_exec_lock()

                return

            # Build task parameter payload from Redis stream data.
            params = {
                "uid": uid,
                "hash": data.get("hash", uid),
                "image": data.get("image", ""),
                "content_type": data.get("content_type", "image/png"),
                "seed": int(data.get("seed", 42)),
                "resolution": str(data.get("resolution", "1024")),
                "decimation_target": int(data.get("decimation_target", 500000)),
                "texture_size": int(data.get("texture_size", 2048)),
                "ss_guidance_strength": float(data.get("ss_guidance_strength", 7.5)),
                "ss_guidance_rescale": float(data.get("ss_guidance_rescale", 0.7)),
                "ss_sampling_steps": int(data.get("ss_sampling_steps", 12)),
                "ss_rescale_t": float(data.get("ss_rescale_t", 5.0)),
                "shape_slat_guidance_strength": float(data.get("shape_slat_guidance_strength", 7.5)),
                "shape_slat_guidance_rescale": float(data.get("shape_slat_guidance_rescale", 0.5)),
                "shape_slat_sampling_steps": int(data.get("shape_slat_sampling_steps", 12)),
                "shape_slat_rescale_t": float(data.get("shape_slat_rescale_t", 3.0)),
                "tex_slat_guidance_strength": float(data.get("tex_slat_guidance_strength", 1.0)),
                "tex_slat_guidance_rescale": float(data.get("tex_slat_guidance_rescale", 0.0)),
                "tex_slat_sampling_steps": int(data.get("tex_slat_sampling_steps", 12)),
                "tex_slat_rescale_t": float(data.get("tex_slat_rescale_t", 3.0)),
            }

            # Flag to track if execution has started, used for retry logic and DLQ handling.
            execution_started = False

            # Flag to track if we lost the execution lock during processing, used to prevent ACKing and allow retries in that case.
            lost_lock_during_execution = False
            
            try:
                if redis_client and lock_key and lock_token:
                    try:
                        current_lock_token = redis_client.get(lock_key)

                    except Exception as exc:
                        logger.warning(f"Failed to verify execution lock for {uid} before processing: {exc}")
                        current_lock_token = None

                    # Double-check lock ownership before running the worker.
                    if current_lock_token != lock_token:
                        logger.warning(
                            f"Skip execution for {uid}: lost execution lock before starting worker - "
                            f"likely being processed by another worker."
                        )
                        return

                # Check terminal status again just before execution in case another worker finished it.
                existing_status = get_status_from_redis(uid)
                if existing_status and is_terminal_status(existing_status):
                    logger.info(f"Skip execution for {uid}: already terminal according to Redis status.")
                    return

                # Increment retry count right before actual execution to accurately track attempts,
                # even if the process hard-crashes (CUDA OOM, segfault) during execution.
                if redis_client:
                    try:
                        new_retry_count = redis_client.incr(retry_key)
                        # Keep retry count metadata for 7 days instead of arbitrary lock-based TTL
                        redis_client.expire(retry_key, 7 * 24 * 3600)
                        logger.info(f"Task {uid} execution attempt {new_retry_count}/{max_retries}.")
                    except Exception as exc:
                        logger.warning(f"Failed to increment retry count for task {uid}: {exc}")

                # Run the worker with the task parameters.
                execution_started = True
                task_start_time = time.time()
                
                # Define the lock heartbeat to renew the execution lock TTL while the task is running.
                heartbeat = LockHeartbeat(lock_key, lock_token)
                heartbeat.start()
                
                try:
                    run_worker(
                        uid=uid,
                        params=params,
                        exec_lock_key=lock_key,
                        exec_lock_token=lock_token,
                    )
                    
                finally:
                    # Stop the heartbeat after execution completes.
                    heartbeat.stop()

                    # Check if the heartbeat detected a lock loss during execution, which means another worker may have reclaimed this task. 
                    # If so, we should not ACK the stream entry to allow for retries.
                    if heartbeat.lock_lost:
                        lost_lock_during_execution = True
                        logger.error(f"Task {uid} lost execution lock during processing. Another worker may have reclaimed it.")

            finally:
                should_ack = False
                ack_successful = False
                lost_ownership = False

                # Get the latest status of the task from Redis.
                status_data = get_status_from_redis(uid)

                # Only acknowledge terminal tasks so non-terminal ones can be retried.
                if status_data:
                    should_ack = is_terminal_status(status_data)

                # Skip ACK if we lost the lock: another worker may have reclaimed and be processing the task.
                if should_ack and lost_lock_during_execution:
                    logger.warning(f"Skip ACK for {uid}: execution lock was lost during processing.")
                    should_ack = False
                    lost_ownership = True

                # Final ownership verification before ACK to handle split-brain scenarios
                # (e.g. brief network partition that caused lock to expire and be re-acquired by another worker).
                if should_ack and redis_client and lock_key and lock_token:
                    try:
                        final_lock_token = redis_client.get(lock_key)
                        
                        if final_lock_token is not None and final_lock_token != lock_token:
                            logger.warning(f"Skip ACK for {uid}: lock ownership changed before ACK (split-brain guard).")
                            should_ack = False
                            lost_ownership = True

                    except Exception as exc:
                        logger.warning(f"Could not verify lock ownership before ACK for {uid}: {exc}")

                try:
                    if should_ack:
                        # Acknowledge the stream entry and clean up only when the task is in a terminal status.
                        ack_and_cleanup_stream_entry(entry_id)

                        # Set the flag to True to allow lock release and prevent retries, since we've successfully ACKed this task as completed/failed.
                        ack_successful = True

                        logger.info(f"Task {uid} acknowledged and stream entry cleaned up ({source}).")
                    else:
                        status_str = status_data.get('status') if isinstance(status_data, dict) else status_data
                        logger.warning(
                            f"Task {uid} not in terminal status after processing. Current status: "
                            f"{status_str if status_data else None}. "
                            f"Stream entry will not be acknowledged to allow for retries."
                        )

                        # Retry count was already incremented before execution if execution_started was True.
                        # We just check if it has exceeded the max retries to move to DLQ.
                        # Do NOT push to DLQ if we lost ownership, since another worker is currently handling it.
                        if execution_started and not lost_ownership:
                            try:
                                current_retry = int(redis_client.get(retry_key) or 0)

                                if current_retry >= max_retries:
                                    logger.error(f"Task {uid} exceeded max retries ({current_retry}/{max_retries}). Moving to dead-letter queue.")

                                    push_to_dlq(uid, data, {
                                        "failed_at": str(time.time()),
                                        "retry_count": str(current_retry),
                                        "consumer_id": str(app_state.consumer_id),
                                        "worker_session_id": worker_session_id,
                                        "hostname": socket.gethostname(),
                                        "execution_time": f"{time.time() - task_start_time:.2f}s",
                                        "reason": "max_retries_exceeded_post_run",
                                    })

                                    set_status_in_redis(uid, {"status": "failed", "error": "Max retries exceeded."})
                                    ack_and_cleanup_stream_entry(entry_id)
                                    ack_successful = True

                            except Exception as exc:
                                logger.warning(f"Failed to check retry count for task {uid}: {exc}")

                except Exception as exc:
                    logger.error(f"Failed to acknowledge and clean up stream entry for task {uid}: {exc}")

                # Release the lock only after a successful ACK.
                # It prevents other workers from picking up this task before we ACK it as completed/failed, which could cause duplicate processing.
                if ack_successful:
                    _release_exec_lock()

                # Explicit CUDA cleanup after each task to prevent VRAM fragmentation.
                aggressive_gpu_cleanup()

            if ack_successful:
                logger.info(f"Task {uid} processed successfully from {source} source.")

        except Exception as exc:
            logger.error(f"Error processing stream entry {entry_id} for task {uid}: {exc}")
            # Release the execution lock when errors occur.
            _release_exec_lock()

    # Flag to check pending tasks on startup and after reconnects.
    check_pending = True

    # Timestamp of the last queue depth reconciliation.
    last_reconcile_ts: float = 0.0

    # Main loop to continuously read from the Redis stream and process tasks.
    while True:
        # Exit the consumer loop if a CUDA OOM has flagged a subprocess restart.
        if app_state.needs_subprocess_restart:
            logger.warning("CUDA OOM recovery: exiting consumer loop to trigger subprocess restart.")
            return

        try:
            # Ensure the consumer group exists before trying to read from the stream.
            if not ensure_consumer_group():
                time.sleep(1)
                continue

            # Skip processing if we're in the middle of draining to allow in-flight tasks to complete and prevent picking up new ones.
            if os.path.exists("/tmp/draining"):
                time.sleep(1)
                continue

            # First, check for tasks already assigned to THIS consumer, or pending list.
            # We do this on startup or after reconnects, rather than every single loop iteration.
            # This helps reduce latency for picking up pending tasks in all scenarios.
            # This also allows us to prioritize processing pending tasks before new ones.
            if check_pending:
                start_pel_id = "0"
                while True:
                    pending_messages = redis_client.xreadgroup(
                        groupname=config.CONSUMER_GROUP,
                        consumername=app_state.consumer_id,
                        streams={config.STREAM_KEY: start_pel_id},
                        count=10,
                    )

                    if not pending_messages:
                        break

                    found_processable = False
                    for _, entries in pending_messages:
                        if not entries:
                            pending_messages = None
                            break
                        for entry_id, data in entries:
                            if data:
                                process_stream_entry(entry_id, data, source="pending")
                                found_processable = True
                            start_pel_id = entry_id

                    if not found_processable or not pending_messages:
                        break
                
                # If no more pending tasks, disable this check until the next connection reset
                check_pending = False

            # Second, attempt to reclaim stale pending tasks left behind by dead workers.
            if xautoclaim_supported:
                try:
                    reclaim_resp = redis_client.xautoclaim(
                        name=config.STREAM_KEY,
                        groupname=config.CONSUMER_GROUP,
                        consumername=app_state.consumer_id,
                        min_idle_time=config.PENDING_RECLAIM_MIN_IDLE_MS,
                        start_id=reclaim_start_id,
                        count=config.PENDING_RECLAIM_BATCH_SIZE,
                    )

                    # Normalize the XAUTOCLAIM response shape across Redis client versions.
                    reclaimed_entries = []
                    if isinstance(reclaim_resp, (list, tuple)):
                        if len(reclaim_resp) >= 2:
                            reclaim_start_id = reclaim_resp[0] or "0-0"
                            reclaimed_entries = reclaim_resp[1] or []
                        else:
                            reclaimed_entries = reclaim_resp[0] or []

                    if reclaimed_entries:
                        logger.warning(f"Reclaimed {len(reclaimed_entries)} stale pending task(s)")

                        # Process reclaimed messages but don't `continue` unconditionally to prevent starvation
                        for entry_id, data in reclaimed_entries:
                            if data:
                                process_stream_entry(entry_id, data, source="reclaimed")
                    else:
                        # Reset the cursor if we've exhausted stale tasks
                        reclaim_start_id = "0-0"

                except Exception as reclaim_err:
                    if "unknown command" in str(reclaim_err).lower():
                        xautoclaim_supported = False
                        logger.warning("XAUTOCLAIM not supported by Redis server. Pending task reclaim will be disabled.")
                    else:
                        logger.warning(f"Error during XAUTOCLAIM for pending tasks: {reclaim_err}")

            # Finally, read new entries from the Redis stream using XREADGROUP.
            messages = redis_client.xreadgroup(
                groupname=config.CONSUMER_GROUP,
                consumername=app_state.consumer_id,
                streams={config.STREAM_KEY: ">"},
                block=1000,
                count=1,
            )

            # If no messages are available, continue waiting.
            if not messages:
                redis_failure_count = 0

                # Reconcile the depth counter during idle periods so any drift accumulated from crashes or out-of-band removals is healed before the next scale decision.
                now = time.time()
                if now - last_reconcile_ts >= config.QUEUE_DEPTH_RECONCILE_INTERVAL_SECONDS:
                    # Reconcile the queue depth counter with the actual stream length to correct any drift.
                    reconcile_queue_depth_counter()
                    
                    last_reconcile_ts = now

                continue

            # Process each stream entry in the messages.
            for _, entries in messages:
                for entry_id, data in entries:
                    process_stream_entry(entry_id, data, source="new")

            # Reset Redis failure count after successful processing
            redis_failure_count = 0

        except Exception as exc:
            # If the consumer group disappeared, invalidate cache and recreate it.
            if "NOGROUP" in str(exc):
                invalidate_consumer_group()
                ensure_consumer_group()

            if is_redis_connection_error(exc):
                redis_failure_count += 1

                # Always re-check PEL, as an interrupted XREADGROUP might have assigned a task to us.
                check_pending = True  

                if redis_failure_count >= 3:
                    # Invalidate the current consumer group
                    invalidate_consumer_group()

                    # Attempt to reset the Redis connection to recover from the error.
                    reset_redis_connection("task consumer loop connection failure")
                    logger.warning(f"Task consumer Redis connectivity issue: {exc}. Resetting connection.")
                else:
                    logger.warning(f"Task consumer Redis transient issue (attempt {redis_failure_count}): {exc}")

                # Wait for a jittered exponential backoff delay before retrying.
                time.sleep(_redis_retry_delay(redis_failure_count))
                continue

            logger.error(f"Task consumer error: {exc}")

            # Sleep before retrying to avoid tight error loops.
            time.sleep(5)
