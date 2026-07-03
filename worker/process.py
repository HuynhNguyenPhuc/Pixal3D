"""Worker process management."""

import multiprocessing as mp
import os
import time

import app_state
import config
from broker.client import redis_client
from broker.state import worker_ready_key
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def start_worker_process() -> bool:
    """
    Start the worker process if not already running.

    Returns:
        True if the worker process started successfully, False otherwise.
    """
    if not app_state.worker_initialization_args:
        logger.error("Worker boot arguments not set. Cannot start worker process.")
        return False

    try:
        # Create a shared readiness event so parent and child agree on worker state.
        app_state.worker_ready_event = mp.Event()
        app_state.worker_ready = False

        # Define the worker subprocess.
        proc = mp.Process(
            target=run_worker_subprocess,
            args=(app_state.worker_initialization_args, app_state.worker_ready_event),
            daemon=True,
        )

        # Start the worker subprocess.
        proc.start()

        # Store the worker process reference in app state.
        with app_state.worker_process_lock:
            app_state.worker_process = proc

        # Store the worker ID in app state for use in other modules.
        app_state.worker_id = f"worker-{proc.pid}"

        logger.warning(f"✅ Worker process started with PID {proc.pid}")
        return True

    except Exception as exc:
        logger.error(f"Failed to start worker process: {exc}")
        return False


def restart_worker_process(reason: str):
    """
    Restart worker process with rate limiting to prevent rapid restarts.

    Args:
        reason (str): The reason for restarting the worker process.
    """
    # Get current time for rate limiting.
    now = time.time()

    while (
        app_state._restart_timestamps
        and now - app_state._restart_timestamps[0] > config.WORKER_RESTART_WINDOW_SECONDS
    ):
        # Remove old timestamps that are outside the restart window.
        app_state._restart_timestamps.popleft()

    # If we have restarted too many times recently, back off before retrying.
    if len(app_state._restart_timestamps) >= config.WORKER_MAX_RESTARTS_PER_WINDOW:
        logger.error(
            f"Restart threshold reached "
            f"({len(app_state._restart_timestamps)}/{config.WORKER_MAX_RESTARTS_PER_WINDOW}). "
            f"Backing off for {config.WORKER_RESTART_BACKOFF_SECONDS}s"
        )

        time.sleep(config.WORKER_RESTART_BACKOFF_SECONDS)
        app_state._restart_timestamps.clear()

    # Get the current worker process.
    with app_state.worker_process_lock:
        proc = app_state.worker_process

    # If the worker process is alive, terminate it before restarting.
    if proc and proc.is_alive():
        try:
            # Terminate the worker process gracefully first.
            proc.terminate()

            # Wait for the process to exit, with a timeout to prevent hanging.
            proc.join(timeout=20)

            if proc.is_alive():
                # Force kill if graceful termination timed out.
                proc.kill()
                proc.join(timeout=5)

        except Exception as exc:
            logger.warning(f"Failed to terminate worker process (PID {proc.pid}): {exc}")

    # Record the restart timestamp for rate limiting.
    app_state._restart_timestamps.append(time.time())

    logger.warning(f"Restarting worker process due to: {reason}")

    # Start a new worker process.
    start_worker_process()


def run_worker_subprocess(worker_args: dict, ready_event):
    """
    The target function for the worker subprocess.

    Args:
        worker_args (dict): The arguments for initializing the worker.
    """
    # Override the SAVE_DIR for the worker subprocess.
    config.SAVE_DIR = worker_args.get("cache_path", config.SAVE_DIR)

    # Ensure the save directory exists.
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    
    # Set worker and consumer IDs for this subprocess.
    # We include the hostname to avoid collisions across multiple pods.
    import socket
    worker_id = f"{socket.gethostname()}-{os.getpid()}"

    app_state.worker_id = worker_id
    app_state.consumer_id = worker_id

    # Reset worker ready state before initialization to ensure accurate health status during startup.
    app_state.worker_ready = False
    if ready_event:
        ready_event.clear()

    # Import the model worker here to avoid circular imports.
    from worker.model import ModelWorker

    # Initialize the worker instance and store it in app state.
    app_state.worker = ModelWorker(
        model_path=worker_args["model_path"],
        device=worker_args["device"],
        worker_id=worker_id,
        save_dir=config.SAVE_DIR,
    )

    # Set the worker ready key in Redis so readiness probes can verify backend health.
    try:
        if redis_client:
            redis_client.set(worker_ready_key(os.getpid()), "1")

    except Exception:
        pass

    # Mark the worker as ready in local state.
    app_state.worker_ready = True
    if ready_event:
        ready_event.set()

    logger.warning(f"✅ Worker subprocess initialized and ready with ID {worker_id}")

    # Import the task consumer daemon here to avoid circular imports.
    from worker.consumer import task_consumer_daemon

    try:
        # Start consuming tasks from the Redis stream.
        task_consumer_daemon()

    finally:
        app_state.worker_ready = False
        if ready_event:
            ready_event.clear()

        # Clean up the readiness marker when the worker subprocess exits.
        try:
            if redis_client:
                redis_client.delete(worker_ready_key(os.getpid()))

        except Exception:
            pass
