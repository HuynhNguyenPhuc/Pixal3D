"""Queue operations for the broker service using Redis Streams."""

import random
import threading
import time
from typing import Optional

import redis as redis_lib

import config
from broker.client import is_redis_connection_error, redis_client, reset_redis_connection
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


# --- Lua Scripts --- #

# Safe DECR: atomically decrement a Redis key but never go below zero.
_LUA_SAFE_DECR = """
local v = redis.call('GET', KEYS[1])
if not v or tonumber(v) <= 0 then
    redis.call('SET', KEYS[1], '0')
    return 0
end
return redis.call('DECR', KEYS[1])
"""


# --- Consumer Group State --- #
consumer_group_lock = threading.Lock()
consumer_group_failures: int = 0
is_consumer_group_ready: bool = False
last_consumer_group_init_time: float = 0.0

# --- Consumer Group Retry Intervals --- #
CONSUMER_GROUP_RETRY_INTERVAL_SUCCESS = 5    # Longer interval when healthy (avoid spam)
CONSUMER_GROUP_RETRY_INTERVAL_FAILURE = 1    # Shorter interval when failed (recover fast)
CONSUMER_GROUP_RETRY_INTERVAL_MAX = 30       # Max backoff window when Redis is unhealthy


def _get_consumer_group_retry_interval() -> float:
    """Return next retry window using exponential backoff with jitter."""
    if is_consumer_group_ready:
        return CONSUMER_GROUP_RETRY_INTERVAL_SUCCESS

    base_interval = min(
        CONSUMER_GROUP_RETRY_INTERVAL_FAILURE * (2 ** min(consumer_group_failures, 5)),
        CONSUMER_GROUP_RETRY_INTERVAL_MAX,
    )
    return base_interval + random.uniform(0.1, 0.9)


# ============================================================================
# Consumer Group Management
# ============================================================================

def ensure_consumer_group() -> bool:
    """
    Ensure the consumer group for the Redis stream is created and ready.

    We use producer-consumer pattern with Redis Streams.
    - The producer writes tasks to the stream.
    - The consumer group reads tasks from the stream.

    Returns:
        True if the consumer group is ready, False otherwise.
    """
    global last_consumer_group_init_time, is_consumer_group_ready, consumer_group_failures

    # If Redis client is not available, we cannot ensure consumer group readiness.
    if not redis_client:
        return False


    # ================================================
    # Fast Path: Check cached state first
    # ================================================

    # Get the current timestamp.
    now = time.time()

    # Determine retry interval based on current state:
    # - If healthy, use longer interval to reduce unnecessary attempts.
    # - If failed, use shorter interval with exponential backoff to recover quickly.
    retry_interval = _get_consumer_group_retry_interval()

    # If we are within the retry interval, return the cached state.
    if now - last_consumer_group_init_time < retry_interval:
        return is_consumer_group_ready


    # ================================================
    # Slow Path: Acquire lock and attempt group creation
    # ================================================

    # Acquire lock to ensure only one thread creates the consumer group at a time.
    with consumer_group_lock:
        # Get the current timestamp again after acquiring the lock.
        now = time.time()

        # Determine retry interval based on current state.
        retry_interval = _get_consumer_group_retry_interval()

        # Re-check cached state after acquiring the lock.
        if now - last_consumer_group_init_time < retry_interval:
            return is_consumer_group_ready

        try:
            # Create the consumer group and stream if needed.
            # Redis natively handles concurrent XGROUP CREATE calls. 
            # One will succeed, the rest will return BUSYGROUP.
            redis_client.xgroup_create(
                name=config.STREAM_KEY,
                groupname=config.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )

            # Mark the consumer group as ready after successful creation.
            is_consumer_group_ready = True
            consumer_group_failures = 0

            logger.info(f"✅ Redis consumer group '{config.CONSUMER_GROUP}' created successfully")
            return True

        except redis_lib.ResponseError as exc:
            # BUSYGROUP means the consumer group already exists, which is still healthy.
            if "BUSYGROUP" in str(exc):
                is_consumer_group_ready = True
                consumer_group_failures = 0
                return True

            # Mark the consumer group as unavailable on any other Redis response error.
            is_consumer_group_ready = False
            consumer_group_failures += 1
            logger.error(f"Failed to create consumer group '{config.CONSUMER_GROUP}': {exc}")
            return False

        except Exception as exc:
            # Mark the consumer group as unavailable on unexpected errors as well.
            is_consumer_group_ready = False
            consumer_group_failures += 1

            # Handle Redis connection errors separately to attempt a reset.
            if is_redis_connection_error(exc):
                # Reset the Redis connection to recover from connectivity issues.
                reset_redis_connection("consumer-group init connection failure")
                logger.warning(f"Failed to create consumer group '{config.CONSUMER_GROUP}' due to Redis connectivity issue: {exc}")
            else:
                logger.exception(f"Failed to create consumer group '{config.CONSUMER_GROUP}': {exc}")
            return False

        finally:
            # Record the last attempt time regardless of success or failure.
            last_consumer_group_init_time = time.time()


def invalidate_consumer_group():
    """
    Invalidate the cached state of the consumer group readiness.

    This should be called when we detect that the consumer group might be unavailable.
    """
    global last_consumer_group_init_time, is_consumer_group_ready, consumer_group_failures

    # Reset the state to force a retry on the next check.
    is_consumer_group_ready = False
    consumer_group_failures = 0

    # Reset the last init timestamp so the next call retries immediately.
    last_consumer_group_init_time = 0.0


# ============================================================================
# Queue Operations
# ============================================================================

def enqueue_task(task_data: dict) -> Optional[str]:
    """
    Enqueue a task to the Redis stream.
    
    Args:
        task_data: Field/value mapping to write as the stream entry payload.

    Returns:
        The stream entry ID on success, or None if Redis is unavailable or the pipeline raises.
    """
    if not redis_client:
        return None

    try:
        # Define a Redis pipeline with transaction flag to ensure atomicity of the stream add and counter increment.
        pipe = redis_client.pipeline(transaction=True)

        # Add the task to the Redis stream.
        pipe.xadd(config.STREAM_KEY, task_data)

        # Increment the queue depth counter.
        pipe.incr(config.QUEUE_DEPTH_KEY)

        # Execute the pipeline and get the entry ID of the added task.
        entry_id, _ = pipe.execute()

        return entry_id

    except Exception as exc:
        if is_redis_connection_error(exc):
            reset_redis_connection("enqueue_task pipeline connection failure")

        logger.error(f"Failed to enqueue task atomically: {exc}")
        return None


def ack_and_cleanup_stream_entry(entry_id: str):
    """
    Acknowledge a Redis stream entry and decrement the depth counter.

    Args:
        entry_id: The Redis stream entry ID to acknowledge and delete.
    """
    if not redis_client:
        return

    # The flag acked indicates whether we successfully acknowledged an entry.
    acked = 0

    # Acknowledge the entry to mark it as processed. This removes it from the pending list of the consumer group.
    try:
        acked = redis_client.xack(
            config.STREAM_KEY, 
            config.CONSUMER_GROUP, 
            entry_id
        )
    
    except Exception as exc:
        logger.warning(f"Failed to ACK stream entry {entry_id}: {exc}")

    # Only attempt to decrement the counter if we actually ACKed an entry (acked > 0), which means it was pending and we are now marking it as completed. 
    # If acked == 0, it means the entry was not pending (e.g. already ACKed or never delivered), so we should not decrement the counter to avoid undercounting.
    if acked:
        try:
            redis_client.eval(_LUA_SAFE_DECR, 1, config.QUEUE_DEPTH_KEY)

        except Exception as exc:
            logger.warning(f"Failed to decrement queue depth counter for entry {entry_id}: {exc}")


def get_queue_position(uid: str) -> Optional[int]:
    """
    Return the approximate position of a task in the queue.

    Args:
        uid: The UID of the task to check.

    Returns:
        The approximate position of the task, or None if it cannot be found.
    """
    if not redis_client:
        return None

    try:
        # Ensure the consumer group is ready before reading the stream.
        ensure_consumer_group()

        # Get recent entries from the stream and scan for the matching UID.
        entries = redis_client.xrange(
            config.STREAM_KEY,
            min="-",
            max="+",
            count=max(config.MAX_QUEUE_DEPTH, 1),
        )

        for idx, (_, data) in enumerate(entries, 1):
            if data.get("uid") == uid:
                return idx

        return None

    except Exception as exc:
        # If the consumer group is missing, we cannot determine queue position.
        if "NOGROUP" in str(exc):
            return None

        logger.warning(f"Failed to get queue position for uid={uid}: {exc}")
        return None


def get_queue_depth() -> int:
    """
    Return the effective queue depth (tasks not yet fully processed).

    Two methods are used to determine the depth:
    1. Primary method: a dedicated Redis counter key that is incremented on enqueue and decremented on ACK. This is the most efficient method and is used as the primary source of truth.
    2. Fallback method: if the counter is unavailable (e.g. Redis restart), we derive the depth from the consumer group metadata (pending + lag) or stream length. 

    Returns:
        Total number of tasks that are pending or yet to be delivered.
    """
    if not redis_client:
        return 0

    # Fast path: dedicated counter is the primary source of truth.
    try:
        # Get the queue depth from the dedicated counter key.
        raw = redis_client.get(config.QUEUE_DEPTH_KEY)
        
        if raw is not None:
            return max(int(raw), 0)
        
    except Exception:
        pass

    # Slow path: derive from consumer-group metadata (used before first enqueue).
    try:
        # Ensure the consumer group is ready before reading the stream.
        ensure_consumer_group()

        # Get consumer-group information for the stream.
        groups = redis_client.xinfo_groups(config.STREAM_KEY)

        for group in groups:
            if group.get("name") != config.CONSUMER_GROUP:
                continue

            # Pending entries have been delivered but not acknowledged yet.
            pending = int(group.get("pending") or 0)

            # Lag entries have not been delivered to any consumer yet.
            lag = int(group.get("lag") or 0)

            return max(pending + lag, 0)

        # If we cannot find the group, fall back to stream length.
        return int(redis_client.xlen(config.STREAM_KEY) or 0)

    except Exception as exc:
        logger.warning(f"Failed to get queue depth: {exc}")

        try:
            # If we got an exception (e.g. consumer group missing), fall back to stream length.
            return int(redis_client.xlen(config.STREAM_KEY) or 0)
        
        except Exception:
            return 0


def remove_task_from_queue(uid: str):
    """
    Best-effort removal of a queued task from the Redis stream.

    Args:
        uid: The UID of the task to remove.
    """
    if not redis_client:
        return

    try:
        # Ensure the consumer group is ready before scanning the stream.
        ensure_consumer_group()

        # Scan a bounded number of entries for the matching UID.
        entries = redis_client.xrange(
            config.STREAM_KEY,
            min="-",
            max="+",
            count=max(config.MAX_QUEUE_DEPTH * 4, 100),
        )

        for entry_id, data in entries:
            if data.get("uid") != uid:
                continue

            try:
                # Acknowledge the entry to remove it from the pending list if it was delivered to a consumer.
                redis_client.xack(
                    config.STREAM_KEY, 
                    config.CONSUMER_GROUP, 
                    entry_id
                )

            except Exception:
                pass

            # Attempt to delete the entry from the stream.  
            # If it was already ACKed and deleted by the consumer, this will be a no-op.
            deleted = redis_client.xdel(config.STREAM_KEY, entry_id)

            
            if deleted:
                try:
                    # DECR only when XDEL actually removed an entry (returns 1).
                    # If deleted == 0 the entry was already gone and whichever path removed it first already decremented the counter.
                    redis_client.eval(_LUA_SAFE_DECR, 1, config.QUEUE_DEPTH_KEY)

                except Exception as exc:
                    logger.warning(f"Failed to decrement queue depth counter for uid={uid}: {exc}")

    except Exception as exc:
        logger.warning(f"Failed to remove task {uid} from queue: {exc}")


def cleanup_old_stream_entries():
    """Remove old stream entries."""
    if not redis_client:
        return

    try:
        # Compute the millisecond timestamp below which entries should be removed.
        # Redis Stream IDs use millisecond Unix timestamps as the first component.
        cutoff_ms = int((time.time() - config.STREAM_RETENTION_SECONDS) * 1000)
        min_id = f"{cutoff_ms}-0"

        # Trim the stream by minimum ID to remove old entries outside of the retention window.
        trimmed = redis_client.xtrim(
            config.STREAM_KEY, 
            minid=min_id, 
            approximate=True
        )

        if trimmed:
            logger.info(f"Stream retention: trimmed {trimmed} entries older than {config.STREAM_RETENTION_SECONDS}s")

    except Exception as exc:
        logger.warning(f"Failed to trim old stream entries: {exc}")


# ============================================================================
# Queue Depth Counter Reconciliation
# ============================================================================

def reconcile_queue_depth_counter() -> bool:
    """
    Reconcile the queue depth counter with the actual stream length.

    Returns:
        True if reconciliation ran successfully (counter may or may not have
        changed), False if ground truth could not be determined.
    """
    if not redis_client:
        return False

    try:
        # Ensure the consumer group is ready before reading the stream.
        ensure_consumer_group()

        # Get all the consumer groups for the stream.
        groups = redis_client.xinfo_groups(config.STREAM_KEY)

        for group in groups:
            if group.get("name") != config.CONSUMER_GROUP:
                continue

            # Get the lag and pending counts from the group info.
            lag_raw = group.get("lag")
            pending = int(group.get("pending") or 0)

            # Get the current counter value.
            current = max(int(redis_client.get(config.QUEUE_DEPTH_KEY) or 0), 0)

            if lag_raw is None:
                if current < pending:
                    # Lag is null (e.g. Redis restarted and stream was trimmed), but we have pending tasks — set the counter to the pending count as a floor to prevent undercount drift.
                    redis_client.set(config.QUEUE_DEPTH_KEY, pending)

                    logger.warning(
                        f"Reconciled queue depth (lag=null): {current} -> {pending} "
                        f"(floor set to pending count)"
                    )

                return True

            # Get the actual lag value by logic.
            actual = max(int(lag_raw) + pending, 0)

            if current != actual:
                # Update the counter to the actual value to correct any drift.
                redis_client.set(config.QUEUE_DEPTH_KEY, actual)

                logger.warning(
                    f"Reconciled queue depth: {current} -> {actual} "
                    f"(lag={lag_raw}, pending={pending})"
                )

            return True

        # Group not found — stream may not exist yet (pre-first-task).
        return False

    except Exception as exc:
        logger.warning(f"Failed to reconcile queue depth counter: {exc}")
        return False


# ============================================================================
# Dead-Letter Queue (DLQ)
# ============================================================================

def push_to_dlq(uid: str, stream_data: dict, extra: dict) -> bool:
    """
    Push a task to the dead-letter queue.

    The dead-letter queue is a separate Redis stream that holds failed tasks for later inspection or reprocessing.

    Args:
        uid: The UID of the failed task.
        stream_data: The original Redis stream payload for the task.
        extra: Additional metadata to attach (e.g. reason, retry_count, hostname).

    Returns:
        True if the entry was inserted, False if it was a duplicate or Redis was unavailable.
    """
    if not redis_client:
        return False

    try:
        dlq_key = f"{config.STREAM_KEY}:deadletter"
        dlq_marker_key = f"task:{uid}:dlq"

        # Deduplicate: only insert if no DLQ marker exists for this task yet.
        if redis_client.set(dlq_marker_key, "1", nx=True, ex=7 * 24 * 3600):
            redis_client.xadd(
                dlq_key, 
                {**stream_data, **extra}, 
                maxlen=1000, 
                approximate=True
            )

            return True

        return False

    except Exception as exc:
        logger.warning(f"Failed to push task {uid} to DLQ: {exc}")
        return False
