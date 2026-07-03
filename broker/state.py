"""Redis state management for tasks and workers."""

import json
from typing import Optional

import config
from broker.client import redis_client
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def running_task_key(uid: str) -> str:
    """
    Generate a Redis key for marking a task as currently running.

    - Key format: Task -> Task UID -> Running attribute
    - Value: A JSON payload containing heartbeat and worker metadata
    """
    return f"task:{uid}:running"


def worker_ready_key(pid: int) -> str:
    """
    Generate a Redis key for marking a worker as ready.

    - Key format: Worker -> Worker PID -> Ready attribute
    - Value: A boolean-like marker indicating the worker is ready to accept tasks
    """
    return f"worker:{pid}:ready"


# ============================================================================
# Task Heartbeat Tracking in Redis
#
# Note:
# - This index tracks all currently running tasks for timeout enforcement.
# - Each worker periodically updates a heartbeat timestamp while a task is active.
# - The watchdog scans this index to find stale tasks that may need recovery.
# ============================================================================

# Note:
# - We use a Redis sorted set (ZSET) to track running tasks and their heartbeats.
# - The task UID is the member, and the last heartbeat timestamp is the score.
# - This allows efficient lookup of stale tasks by score range.

def record_task_heartbeat(uid: str, heartbeat_ts: float):
    """
    Record the current heartbeat timestamp for a running task in Redis.

    Args:
        uid: The unique identifier of the task.
        heartbeat_ts: The timestamp of the last heartbeat for the running task.
    """
    if not redis_client:
        return

    try:
        # Add or update the task UID in the sorted-set heartbeat index.
        redis_client.zadd(config.ACTIVE_TASK_HEARTBEAT_KEY, {uid: float(heartbeat_ts)})

    except Exception as exc:
        logger.warning(f"Failed to update running task index for uid={uid}: {exc}")


def remove_running_task(uid: str):
    """
    Remove a running task from Redis heartbeat tracking.

    Args:
        uid: The unique identifier of the task.
    """
    if not redis_client:
        return

    try:
        # Remove the task UID from the sorted-set heartbeat index.
        redis_client.zrem(config.ACTIVE_TASK_HEARTBEAT_KEY, uid)

        # Remove the detailed running-task marker as well.
        redis_client.delete(running_task_key(uid))

    except Exception as exc:
        logger.warning(f"Failed to remove running task index for uid={uid}: {exc}")


def get_running_task_count() -> int:
    """
    Return the number of running tasks tracked in Redis.

    Returns:
        Number of running tasks in the heartbeat index.
    """
    if not redis_client:
        return 0

    try:
        return int(redis_client.zcard(config.ACTIVE_TASK_HEARTBEAT_KEY) or 0)
    except Exception:
        return 0


def task_status_key(uid: str) -> str:
    """
    Generate a Redis key for storing the status of a task.

    - Key format: Task -> Task UID -> Status attribute
    - Value: A JSON string containing task status information
    """
    return f"task:{uid}:status"


# ============================================================================
# Task Status Management
# ============================================================================

def get_status_from_redis(uid: str) -> Optional[dict]:
    """
    Retrieve the status of a task from Redis.

    Args:
        uid: The unique identifier of the task.

    Returns:
        A dictionary containing the status information of the task, or None.
    """
    if not redis_client:
        return None

    try:
        # Get the raw JSON string from Redis.
        raw = redis_client.get(task_status_key(uid))

        if not raw:
            return None

        # Parse the JSON string into a dictionary.
        return json.loads(raw)

    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse status JSON for uid={uid}: {exc}")
        return None

    except Exception as exc:
        logger.warning(f"Failed to get status from Redis for uid={uid}: {exc}")
        return None


def set_status_in_redis(uid: str, status_data: dict, ttl: int = None, force: bool = False):
    """
    Write status JSON to Redis with TTL.

    Args:
        uid: The unique identifier of the task.
        status_data: A dictionary containing the status information of the task.
        ttl: Time-to-live for the Redis key in seconds. If None, uses config default.
        force: If True, bypass timeout-overwrite protection.
    """
    if not redis_client:
        return

    # If not forcing, refuse to overwrite a timeout failure.
    if not force:
        existing_status = get_status_from_redis(uid)

        if is_timeout_failure(existing_status):
            logger.warning(
                f"Task {uid} has already timed out. Refusing to overwrite timeout status "
                f"with new status: {status_data.get('status')}"
            )
            return

    # If no TTL is provided, use the configured default.
    if ttl is None:
        ttl = config.STATUS_TTL

    try:
        # Write the status payload to Redis as JSON.
        redis_client.set(task_status_key(uid), json.dumps(status_data), ex=ttl)
        logger.debug(f"Status updated for uid={uid}: {status_data.get('status')}")

    except Exception as exc:
        logger.warning(f"Failed to set status in Redis for uid={uid}: {exc}")


def is_terminal_status(status_data: Optional[dict]) -> bool:
    """
    Check if the status indicates a terminal state.

    Args:
        status_data: A dictionary containing the status information of the task.

    Returns:
        True if the status is terminal, False otherwise.
    """
    if not status_data:
        return False

    status = status_data.get("status")
    
    # If it's an error but marked as retriable, it's not terminal from the queue's perspective.
    # We want it to be retried (via PEL/XAUTOCLAIM) rather than ACKed and discarded.
    if status == "error" and status_data.get("retriable") is True:
        return False
        
    # Consider "completed", "error" (non-retriable), "cancelled", "interrupted", and "failed" as terminal states.
    return status in {"completed", "error", "cancelled", "interrupted", "failed"}


def is_timeout_failure(status_data: Optional[dict]) -> bool:
    """
    Check if the status indicates a failure due to execution timeout.

    Args:
        status_data: A dictionary containing the status information of the task.

    Returns:
        True if the status indicates a timeout failure, False otherwise.
    """
    if not status_data:
        return False

    return (
        status_data.get("status") == "error"
        and status_data.get("error_code") == "EXECUTION_TIMEOUT"
    )


def has_task_timed_out(uid: str) -> bool:
    """
    Check if a task has already been marked as timed out.

    Args:
        uid: The unique identifier of the task.

    Returns:
        True if the task has timed out, False otherwise.
    """
    return is_timeout_failure(get_status_from_redis(uid))
