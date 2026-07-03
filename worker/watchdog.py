"""Worker watchdog daemon to monitor worker process and enforce execution timeout."""

import json
import time

import app_state
import config
from broker.client import is_redis_connection_error, redis_client, reset_redis_connection
from broker.state import (
    get_status_from_redis,
    is_terminal_status,
    record_task_heartbeat,
    remove_running_task,
    running_task_key,
    set_status_in_redis,
)
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def _retrieve_stale_running_tasks(now_ts: float):
    """
    Retrieve running tasks that have stale heartbeats, indicating potential execution timeout.

    Args:
        now_ts (float): Current timestamp to compare against heartbeat timestamps.

    Yields:
        Tuple[str, str, float, float, Optional[int]]: Task marker key, uid, started_at,
        last_heartbeat, and worker pid.
    """
    if not redis_client:
        return

    # Calculate the cutoff timestamp for stale heartbeats.
    stale_before = now_ts - (config.EXECUTION_TIMEOUT_SECONDS + config.EXECUTION_TIMEOUT_GRACE_SECONDS)

    try:
        # Get all the UIDs of running tasks that are already stale.
        stale_uids = redis_client.zrangebyscore(
            name=config.ACTIVE_TASK_HEARTBEAT_KEY,
            min="-inf",
            max=stale_before,
        )

        for stale_uid in stale_uids:
            # Get the running task key from the UID.
            key = running_task_key(stale_uid)

            # Retrieve the raw JSON payload for the running task.
            raw = redis_client.get(key)

            if not raw:
                # If the key is already gone, remove the stale heartbeat index entry.
                remove_running_task(stale_uid)
                continue

            try:
                # Parse the running-task payload.
                data = json.loads(raw)

            except Exception:
                logger.warning(f"Invalid running task data for UID {stale_uid}: {raw}")
                remove_running_task(stale_uid)
                continue

            # Extract the task metadata from the payload.
            uid = data.get("uid") or stale_uid
            started_at = data.get("started_at")
            last_heartbeat = data.get("last_heartbeat", started_at)
            pid = data.get("pid")

            # If critical metadata is missing, remove the stale entry.
            if not uid or started_at is None:
                remove_running_task(stale_uid)
                continue

            # If heartbeat is no longer stale, refresh its score and skip it.
            if float(last_heartbeat or started_at) > stale_before:
                record_task_heartbeat(uid, float(last_heartbeat or started_at))
                continue

            yield key, uid, float(started_at), float(last_heartbeat or started_at), pid

    except Exception as exc:
        # Handle Redis connection errors separately to attempt a reset.
        if is_redis_connection_error(exc):
            # Reset the Redis connection to recover from connectivity issues.
            reset_redis_connection("watchdog stale task scan connection failure")
            logger.warning(f"Failed to iterate stale running tasks due to Redis connectivity issue: {exc}")
            return

        logger.warning(f"Failed to iterate stale running tasks: {exc}")


def run_watchdog_sweep(current_worker_pid: int, restart_worker_process, now_ts: float | None = None) -> dict:
    """
    Run one watchdog sweep over stale running tasks.

    Args:
        current_worker_pid (int): PID of the currently live worker process.
        restart_worker_process: Callback used when timeout recovery requires a restart.
        now_ts (float | None): Optional timestamp override for tests.

    Returns:
        dict: Summary of what the sweep cleaned or escalated.
    """
    if now_ts is None:
        now_ts = time.time()

    sweep_result = {
        "terminal_cleanup_count": 0,
        "timed_out_uid": None,
        "restart_reason": None,
    }

    for key, uid, started_at, last_heartbeat, pid in (_retrieve_stale_running_tasks(now_ts) or []):
        # Ignore stale markers that belong to an older worker process. The live
        # process should not time out work it does not own.
        if pid and int(pid) != int(current_worker_pid):
            continue

        # Re-read the task status before escalating. A stale running marker may
        # survive briefly after the real terminal status has already been written.
        current_status = get_status_from_redis(uid)

        if current_status and is_terminal_status(current_status):
            try:
                if redis_client:
                    redis_client.delete(key)

                remove_running_task(uid)
                sweep_result["terminal_cleanup_count"] += 1

                logger.info(
                    f"Cleaned stale running marker for terminal task {uid} "
                    f"({current_status.get('status')})"
                )

            except Exception:
                logger.warning(f"Failed to clean up terminal task {uid} in watchdog.")
                pass

            continue

        # Use the freshest activity timestamp we have. A task can remain valid if
        # either its start time or heartbeat was updated recently enough.
        activity_at = max(started_at, last_heartbeat)
        elapsed = now_ts - activity_at

        if elapsed <= (config.EXECUTION_TIMEOUT_SECONDS + config.EXECUTION_TIMEOUT_GRACE_SECONDS):
            continue

        logger.error(
            f"Execution timeout exceeded for task {uid}: "
            f"{elapsed:.1f}s > {config.EXECUTION_TIMEOUT_SECONDS}s"
        )

        timeout_worker_id = f"worker-{pid or current_worker_pid}"

        # Persist the timeout as the new terminal state before removing markers
        # or restarting the subprocess.
        set_status_in_redis(
            uid,
            {
                "status": "error",
                "hash": uid,
                "message": "Task timeout.",
                "detail": f"Exceeded execution timeout of {config.EXECUTION_TIMEOUT_SECONDS}s.",
                "error_code": "EXECUTION_TIMEOUT",
                "error_type": "internal",
                "retriable": True,
                "retry_after": 0,
                "worker_id": timeout_worker_id,
            },
            force=True,
        )

        if redis_client:
            try:
                redis_client.delete(key)

            except Exception:
                logger.warning(f"Failed to delete running task key {key} for timed-out task {uid}.")
                pass

        remove_running_task(uid)

        restart_reason = f"Execution timeout for task {uid}."
        sweep_result["timed_out_uid"] = uid
        sweep_result["restart_reason"] = restart_reason

        restart_worker_process(restart_reason)
        break

    return sweep_result


def worker_process_watchdog() -> None:
    """
    Worker process watchdog to monitor backend health and enforce execution timeouts.

    The watchdog performs two main duties:
    1. Restart the worker process if it is no longer alive.
    2. Mark stale running tasks as timed out and restart the worker backend.
    """
    logger.info("🩺 Worker process watchdog started!")

    # Import here to avoid circular import issues.
    from worker.process import restart_worker_process

    while True:
        try:
            # Acquire the worker process lock to safely access the worker process.
            with app_state.worker_process_lock:
                proc = app_state.worker_process

            # Restart the worker process if it is not alive.
            if not proc or not proc.is_alive():
                restart_worker_process("Worker process is not alive.")
                time.sleep(config.WORKER_WATCHDOG_INTERVAL_SECONDS)
                continue

            # Run one sweep over stale tasks. The helper is factored out so the
            # exact terminal-cleanup and timeout paths can be smoke-tested.
            run_watchdog_sweep(proc.pid, restart_worker_process, now_ts=time.time())

        except Exception as exc:
            logger.error(f"Worker watchdog encountered an error: {exc}")

        # Watchdog interval.
        time.sleep(config.WORKER_WATCHDOG_INTERVAL_SECONDS)
