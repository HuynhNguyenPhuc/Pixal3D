"""Signal handlers for graceful shutdown in subprocess worker mode."""

import signal
import sys

import app_state
import config
from broker.client import redis_client
from broker.state import remove_running_task, set_status_in_redis
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def graceful_shutdown_handler(signum=None, frame=None) -> None:
    """Interrupt active tasks and stop the worker subprocess.

    Args:
        signum: Signal number supplied by the runtime.
        frame: Current execution frame supplied by the runtime.
    """
    del frame

    # Announce shutdown once before we start mutating task state so the log
    # clearly marks when the runtime began draining active work.
    logger.warning(
        f"Received signal {signum}, initiating graceful shutdown ({config.SHUTDOWN_GRACE_PERIOD}s grace period)..."
    )

    # Read the active heartbeat index first. The worker may already be exiting,
    # but any task still listed here should be marked interrupted instead of
    # being left behind as indefinitely running.
    active_uids = []
    if redis_client:
        try:
            active_uids = redis_client.zrange(config.ACTIVE_TASK_HEARTBEAT_KEY, 0, -1) or []
        except Exception as exc:
            logger.warning(f"Failed to read active tasks during shutdown: {exc}")

    for uid in active_uids:
        # Write terminal interrupted status before cleaning running markers so
        # external status polls see an explicit shutdown outcome.
        set_status_in_redis(
            uid,
            {
                "status": "interrupted",
                "hash": uid,
                "message": "Server shutdown during processing",
                "retriable": True,
                "worker_id": app_state.worker_id,
            },
            force=True,
        )
        remove_running_task(uid)

    # Stop the subprocess worker after task state has been updated.
    with app_state.worker_process_lock:
        proc = app_state.worker_process

    if proc and proc.is_alive():
        proc.terminate()
        proc.join(timeout=config.SHUTDOWN_GRACE_PERIOD)

        if proc.is_alive():
            # Escalate only if graceful termination did not complete in time.
            proc.kill()
            proc.join(timeout=5)

    logger.warning("Graceful shutdown complete. Exiting.")
    sys.exit(0)


def register_signal_handlers() -> None:
    """
    Register SIGTERM and SIGINT handlers.

    The runtime treats both paths the same so shutdown behavior stays
    consistent across local interrupts and orchestrator-driven termination.
    """
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)
    signal.signal(signal.SIGINT, graceful_shutdown_handler)
