"""Locking utilities for task execution."""

import threading
from typing import Optional, Any

import config
from broker.client import redis_client
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)

# Lua script for atomic compare-and-delete
# It releases the lock only when the stored token still matches the caller's token. 
# The GET + DEL approach has a TOCTOU race where another worker can claim the lock in between.
_RELEASE_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

# Lua script for atomic compare-and-expire
# It refreshes TTL only when the caller still owns the lock. 
# The GET + EXPIRE approach has a TOCTOU race where another worker can claim the lock in between.
_RENEW_LOCK_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
else
    return 0
end
"""

# Initialize Lua scripts at module load time.
_release_script_obj: Optional[Any] = None
_renew_script_obj: Optional[Any] = None

# Register Lua scripts if Redis client is available at module load time.
if redis_client:
    try:
        _release_script_obj = redis_client.register_script(_RELEASE_LOCK_SCRIPT)
        _renew_script_obj = redis_client.register_script(_RENEW_LOCK_SCRIPT)

    except Exception as exc:
        logger.warning(f"Failed to register Lua scripts on module load: {exc}")


def execution_lock_key(uid: str) -> str:
    """
    Generate a Redis key for the execution lock of a task.

    - Key format: Task -> Task UID -> Execution Lock attribute
    - Value: A unique token representing the owner of the lock
    """
    return f"task:{uid}:exec_lock"


def acquire_exec_lock(lock_key: str, lock_token: str) -> bool:
    """
    Acquire lock for task execution.

    Args:
        lock_key: The Redis key for the lock.
        lock_token: A unique token that identifies the owner of the lock.

    Returns:
        True if lock acquired successfully, False otherwise.
    """
    # If Redis is unavailable, the runtime cannot make a safe ownership claim.
    # Return False so callers skip execution instead of risking duplicate work.
    if not redis_client:
        return False

    try:
        # Use NX + TTL so only one worker can claim execution at a time and the
        # lock eventually expires if a worker dies mid-task.
        return bool(
            redis_client.set(
                lock_key,
                lock_token,
                nx=True,
                ex=config.EXEC_LOCK_TTL_SECONDS,
            )
        )

    except Exception as exc:
        logger.warning(f"Failed to acquire lock for {lock_key}: {exc}")
        return False


def release_exec_lock(lock_key: str, lock_token: str) -> bool:
    """
    Release lock only if we still own it (atomic CAS via Lua).

    Args:
        lock_key: The Redis key for the lock.
        lock_token: The token that identifies our ownership of the lock.

    Returns:
        True if the lock was released successfully, False if ownership was lost or an error occurred.
    """
    if not redis_client or not lock_key or not lock_token:
        return False

    # Register the Lua script if it wasn't registered at module load time.
    global _release_script_obj
    if _release_script_obj is None and redis_client:
        try:
            _release_script_obj = redis_client.register_script(_RELEASE_LOCK_SCRIPT)
        
        except Exception as exc:
            logger.warning(f"Failed to register release Lua script: {exc}")
            return False

    try:
        # The Lua script will check if the current value of the lock key matches the provided token.
        # If it matches, it deletes the key and returns 1. 
        # If it doesn't match, it returns 0.
        result = _release_script_obj(
            keys=[lock_key], 
            args=[lock_token]
        )

        return bool(result)
    
    except Exception as exc:
        logger.warning(f"Failed to release lock for {lock_key}: {exc}")
        return False


def renew_exec_lock(lock_key: str, lock_token: str) -> bool:
    """
    Refresh TTL on execution lock if we still own it (atomic CAS via Lua).

    Args:
        lock_key: The Redis key for the lock.
        lock_token: The token that identifies our ownership of the lock.

    Returns:
        True if the lock was renewed successfully, False if ownership was lost or an error occurred.
    """
    if not redis_client or not lock_key or not lock_token:
        return False

    # Register the Lua script if it wasn't registered at module load time.
    global _renew_script_obj
    if _renew_script_obj is None and redis_client:
        try:
            _renew_script_obj = redis_client.register_script(_RENEW_LOCK_SCRIPT)
        
        except Exception as exc:
            logger.warning(f"Failed to register renew Lua script: {exc}")
            return False

    try:
        # The Lua script will check if the current value of the lock key matches the provided token.
        # If it matches, it updates the TTL of the key and returns 1.
        # If it doesn't match, it returns 0.
        result = _renew_script_obj(
            keys=[lock_key], 
            args=[lock_token, config.EXEC_LOCK_TTL_SECONDS]
        )

        return bool(result)

    except Exception as exc:
        logger.warning(f"Failed to renew lock for {lock_key}: {exc}")
        return False


def check_lock_ownership(lock_key: str, lock_token: str) -> bool:
    """
    Check if we currently own the lock.

    Args:
        lock_key: The Redis key for the lock.
        lock_token: The token that identifies our ownership of the lock.

    Returns:
        True if we own the lock, False otherwise.
    """
    if not redis_client or not lock_key or not lock_token:
        return False

    try:
        # Get the current value of the lock key.
        value = redis_client.get(lock_key)
            
        # Check if the value matches our token. 
        # If it does, we still own the lock.
        # If it doesn't, we lost ownership.
        return value == lock_token

    except Exception as exc:
        logger.warning(f"Failed to check lock ownership for {lock_key}: {exc}")
        return False


class LockHeartbeat:
    """Background thread to renew execution lock TTL while task is running."""
    def __init__(self, lock_key: str, lock_token: str):
        self.lock_key = lock_key
        self.lock_token = lock_token

        # Event to signal the heartbeat thread to stop when the task is done.
        self._stop_event = threading.Event()

        # Placeholder for the heartbeat thread.
        self._thread = None

        # Flag to indicate if the lock has been lost. 
        # The main task can check this flag to decide whether to continue or abort execution.
        self.lock_lost = False

    def start(self):
        # Define the heartbeat thread to renew the lock TTL periodically.
        self._thread = threading.Thread(
            target=self._heartbeat, 
            daemon=True
        )

        # Start the heartbeat thread.
        self._thread.start()

    def stop(self):
        # Signal the heartbeat thread to stop and wait for it to finish.
        self._stop_event.set()

        # Wait for the thread to finish with a timeout to avoid hanging indefinitely.
        if self._thread:
            self._thread.join(timeout=2.0)

    def _heartbeat(self):
        # Refresh interval: 1/3 of TTL to ensure we renew before expiration
        interval = max(1.0, config.EXEC_LOCK_TTL_SECONDS / 3.0)

        # Number of consecutive failures to renew the lock.
        _consecutive_failures = 0

        while not self._stop_event.is_set():
            # Wait for the specified interval or until stop signal is set.
            if self._stop_event.wait(interval):
                break

            try:
                # Attempt to renew the lock TTL. 
                # If it returns False, we lost ownership.
                success = renew_exec_lock(self.lock_key, self.lock_token)

                if not success:
                    self.lock_lost = True
                    break

                # Reset the consecutive failure count.
                _consecutive_failures = 0

            except Exception as exc:
                # Increment the consecutive failure count.
                _consecutive_failures += 1

                # Log first failure and every 10th thereafter to remain visible without spamming.
                if _consecutive_failures <= 3 or _consecutive_failures % 10 == 0:
                    logger.warning(f"Lock heartbeat renew error for {self.lock_key} (failure #{_consecutive_failures}): {exc}")
