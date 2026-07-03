"""Health controller for Pixal3D API server."""

from fastapi.responses import JSONResponse

import app_state
import config
from broker.client import redis_client
from broker.queue import get_queue_depth
from broker.state import get_running_task_count
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


async def handle_health() -> JSONResponse:
    """Liveness probe."""
    if app_state.run_mode == "api":
        # If running in API mode, report Redis reachability only.
        redis_ok = redis_client is not None

        return JSONResponse(
            {
                "status": "healthy",
                "node_id": app_state.node_id,
                "worker_status": "none (API mode)",
                "redis": "ok" if redis_ok else "unavailable",
            },
            status_code=200,
        )

    with app_state.worker_process_lock:
        proc = app_state.worker_process

    worker_status = f"process:{proc.pid}" if proc and proc.is_alive() else "process:down"

    return JSONResponse(
        {
            "status": "healthy",
            "node_id": app_state.node_id,
            "worker_id": app_state.worker_id,
            "worker_status": worker_status,
        },
        status_code=200,
    )


async def handle_ready() -> JSONResponse:
    """Readiness probe."""
    if app_state.run_mode == "api":
        # If running in API mode, report Redis reachability only.
        if not redis_client:
            return JSONResponse(
                {
                    "status": "not_ready", 
                    "reason": "Redis unavailable"
                },
                status_code=503,
            )
        
        return JSONResponse(
            {
                "status": "ready", 
                "node_id": app_state.node_id, 
                "mode": "api"
            },
            status_code=200,
        )

    with app_state.worker_process_lock:
        proc = app_state.worker_process

    if not proc or not proc.is_alive():
        return JSONResponse(
            {
                "status": "not_ready", 
                "reason": "Worker process unavailable"
            },
            status_code=503,
        )

    # Read readiness from a shared cross-process event, with bool fallback.
    ready_event = app_state.worker_ready_event
    is_ready = bool(ready_event and ready_event.is_set())

    if not is_ready and app_state.worker_ready:
        is_ready = True

    if not is_ready:
        return JSONResponse(
            {
                "status": "not_ready", 
                "reason": "Worker process warming up"
            },
            status_code=503,
        )

    return JSONResponse(
        {
            "status": "ready", 
            "worker_id": app_state.worker_id, 
            "node_id": app_state.node_id
        },
        status_code=200,
    )


async def handle_load() -> JSONResponse:
    """Return current queue and worker load."""
    if app_state.run_mode == "api":
        # If running in API mode, report Redis reachability and queue status only.
        return JSONResponse(
            {
                "status": "ready",
                "node_id": app_state.node_id,
                "mode": "api",
                "queue_depth": get_queue_depth(),
                "queue_capacity": config.MAX_QUEUE_DEPTH,
            },
            status_code=200,
        )

    with app_state.worker_process_lock:
        proc = app_state.worker_process

    if not proc or not proc.is_alive():
        return JSONResponse({"status": "not_ready"}, status_code=503)

    return JSONResponse(
        {
            "status": "ready",
            "worker_id": app_state.worker_id,
            "running_tasks": get_running_task_count(),
            "queue_depth": get_queue_depth(),
            "queue_capacity": config.MAX_QUEUE_DEPTH,
        },
        status_code=200,
    )


async def handle_scaler_depth() -> JSONResponse:
    """Get the current queue depth for KEDA scaling decisions."""
    depth = get_queue_depth()

    return JSONResponse(
        {"depth": depth}, 
        status_code=200
    )
