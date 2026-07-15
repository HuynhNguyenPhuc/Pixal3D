"""Generation controller for Pixal3D API server."""

from fastapi.responses import JSONResponse

import app_state
import config
from broker.client import redis_client
from broker.queue import enqueue_task, get_queue_depth, get_queue_position, remove_task_from_queue
from broker.state import get_status_from_redis, running_task_key, set_status_in_redis
from schemas.generation import GenerationRequest
from utilities.image import get_image_size
from utilities.logger import get_logger
from utilities.validation import validate_gcs_url, validate_http_url, validate_uid_format


# --- Logger --- #
logger = get_logger(__name__)


def build_generation_params(request: GenerationRequest) -> dict:
    """
    Build the worker payload from the validated request.

    Args:
        request (GenerationRequest): Validated API request payload.

    Returns:
        dict: Worker payload to be written to the Redis stream.
    """
    return {
        "uid": request.hash,
        "hash": request.hash,
        "image": request.image,
        "content_type": request.content_type,
        "seed": request.seed,
        "decimation_target": request.decimation_target,
        "texture_size": request.texture_size,
        "ss_guidance_strength": request.ss_guidance_strength,
        "ss_guidance_rescale": request.ss_guidance_rescale,
        "ss_sampling_steps": request.ss_sampling_steps,
        "ss_rescale_t": request.ss_rescale_t,
        "shape_slat_guidance_strength": request.shape_slat_guidance_strength,
        "shape_slat_guidance_rescale": request.shape_slat_guidance_rescale,
        "shape_slat_sampling_steps": request.shape_slat_sampling_steps,
        "shape_slat_rescale_t": request.shape_slat_rescale_t,
        "tex_slat_guidance_strength": request.tex_slat_guidance_strength,
        "tex_slat_guidance_rescale": request.tex_slat_guidance_rescale,
        "tex_slat_sampling_steps": request.tex_slat_sampling_steps,
        "tex_slat_rescale_t": request.tex_slat_rescale_t,
        "fov": request.fov,
        "no_webp": str(request.no_webp).lower(),
    }


async def handle_send(request: GenerationRequest) -> JSONResponse:
    """
    Submit a new generation task to the Redis Stream queue.

    Args:
        request (GenerationRequest): Validated request body.

    Returns:
        JSONResponse: Queueing result payload.
    """
    uid = request.hash

    # Validate task identity.
    is_valid, error_msg = validate_uid_format(uid)

    if not is_valid:
        logger.warning(f"Invalid UID format received: {error_msg}")
        return JSONResponse(status_code=400, content={"error": error_msg, "hash": uid})

    # Validate image source.
    image_source = request.image

    if not image_source:
        return JSONResponse(status_code=400, content={"error": "Missing image source", "hash": uid})

    if not validate_gcs_url(image_source) and not validate_http_url(image_source):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid image source. Image must be a valid gs:// GCS URL or http(s) URL.",
                "hash": uid,
            },
        )

    # Validate Redis availability before attempting to queue the task.
    if not redis_client:
        logger.error("Redis client not initialized.")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "hash": uid, "error": "Redis unavailable for task queueing"},
        )

    try:
        # Enforce payload size before queueing.
        image_size = get_image_size(image_source)

        if image_size > config.MAX_IMAGE_SIZE:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Image size {image_size} exceeds maximum allowed size {config.MAX_IMAGE_SIZE}",
                    "hash": uid,
                },
            )

        # Check if the task is already active (queued, running, or uploading) to prevent duplicate stream entries
        existing_status = get_status_from_redis(uid)
        if existing_status:
            status_str = existing_status.get("status")
            if status_str in {"queued", "running", "uploading"}:
                logger.info(f"Task {uid} is already active with status '{status_str}'. Returning existing status instead of queueing duplicate.")
                
                # Check if it has a queue position
                q_pos = get_queue_position(uid) if status_str == "queued" else None
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": status_str,
                        "hash": uid,
                        "queue_position": q_pos,
                        "message": "Task is already active in queue or running.",
                    },
                )

        # Reject when the stream backlog is already at capacity.
        queue_depth = get_queue_depth()

        if queue_depth >= config.MAX_QUEUE_DEPTH:
            logger.warning(f"Queue full, rejecting task {uid}")

            return JSONResponse(
                status_code=429,
                content={
                    "status": "queue_full",
                    "hash": uid,
                    "error": f"Server queue is full (max {config.MAX_QUEUE_DEPTH}). Please retry later.",
                    "retry_after": 10,
                },
            )

        # Set initial status in Redis first to prevent race conditions with workers picking up the task too quickly.
        set_status_in_redis(
            uid,
            {
                "status": "queued",
                "hash": uid,
                "queue_position": queue_depth + 1,
                "worker_id": app_state.worker_id,
            },
            force=True,
        )

        # Reset retry counter for manual user retry/resubmission of the same hash.
        try:
            redis_client.delete(f"task:{uid}:retries")
        except Exception as exc:
            logger.warning(f"Failed to reset retry counter for task {uid}: {exc}")

        # Enqueue the task into the Redis stream and increment the depth counter atomically.
        task_data = build_generation_params(request)
        entry_id = enqueue_task(task_data)

        if not entry_id:
            logger.error(f"Failed to enqueue task {uid}: Redis pipeline returned no entry ID")
            # If enqueuing failed, set status back to error.
            set_status_in_redis(
                uid,
                {
                    "status": "error",
                    "hash": uid,
                    "message": "Failed to enqueue task",
                },
                force=True,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error", 
                    "hash": uid, 
                    "error": "Failed to enqueue task"
                },
            )

        logger.info(f"Task {uid} added to stream with entry ID {entry_id}. Queue depth: {queue_depth + 1}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "queued",
                "hash": uid,
                "queue_position": queue_depth + 1,
                "message": "Task submitted for processing",
            },
        )

    except Exception as exc:
        logger.error(f"Failed to queue task {uid}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "hash": uid, "error": str(exc)},
        )


async def handle_status(uid: str) -> JSONResponse:
    """
    Check the status of a generation task using Redis only.

    Args:
        uid (str): Task identifier.

    Returns:
        JSONResponse: Redis-backed status payload for the requested task.
    """
    is_valid, error_msg = validate_uid_format(uid)

    if not is_valid:
        logger.warning(f"Invalid UID format in /status: {error_msg}")
        return JSONResponse(
            status_code=400,
            content={"status": "invalid_uid", "hash": uid, "error": error_msg},
        )

    # Status handling is Redis-backed only.
    if not redis_client:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "hash": uid, "message": "Redis unavailable for status lookup"},
        )

    try:
        # Get the current status of the task from Redis.
        status_data = get_status_from_redis(uid)

        if status_data:
            task_status = status_data.get("status")

            if task_status == "completed":
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "completed",
                        "hash": uid,
                        "filename": status_data.get("filename"),
                        "url": status_data.get("url"),
                    },
                )

            if task_status == "error":
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "error",
                        "hash": uid,
                        "message": status_data.get("message", "Unknown error"),
                        "detail": status_data.get("detail", ""),
                        "error_code": status_data.get("error_code"),
                        "error_type": status_data.get("error_type"),
                        "retriable": status_data.get("retriable", False),
                    },
                )

            if task_status == "interrupted":
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "interrupted",
                        "hash": uid,
                        "message": status_data.get("message", "Task interrupted due to server shutdown"),
                        "retriable": status_data.get("retriable", True),
                    },
                )

            if task_status == "cancelled":
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "cancelled",
                        "hash": uid,
                        "message": status_data.get("message", "Task cancelled by user"),
                    },
                )

            # For queued tasks, Redis stream position is still useful metadata.
            queue_position = get_queue_position(uid) if task_status == "queued" else None
            return JSONResponse(
                status_code=200,
                content={"status": task_status, "hash": uid, "queue_position": queue_position},
            )

        # If there is no status payload yet, inspect queue and running markers from Redis.
        queue_position = get_queue_position(uid)
        if queue_position is not None:
            return JSONResponse(
                status_code=200,
                content={"status": "queued", "hash": uid, "queue_position": queue_position},
            )

        if redis_client.exists(running_task_key(uid)):
            return JSONResponse(
                status_code=200,
                content={"status": "running", "hash": uid},
            )

        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "hash": uid, "message": "Task not found!"},
        )

    except Exception as exc:
        logger.error(f"Status check failed for {uid}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "hash": uid, "message": str(exc)},
        )


async def handle_cancel(uid: str) -> JSONResponse:
    """
    Cancel a queued or running task.

    Args:
        uid (str): Task identifier.

    Returns:
        JSONResponse: Cancellation result payload.
    """
    is_valid, error_msg = validate_uid_format(uid)

    if not is_valid:
        logger.warning(f"Invalid UID format in /cancel: {error_msg}")
        return JSONResponse(
            status_code=400,
            content={"status": "invalid_uid", "hash": uid, "error": error_msg},
        )

    if not redis_client:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "hash": uid, "error": "Redis unavailable for task cancellation"},
        )

    try:
        # Resolve the current status before mutating queue state.
        status_data = get_status_from_redis(uid)

        if not status_data:
            return JSONResponse(
                status_code=404,
                content={"status": "not_found", "hash": uid, "error": "Task not found"},
            )

        task_status = status_data.get("status")

        if task_status == "completed":
            return JSONResponse(
                status_code=400,
                content={"status": "already_completed", "hash": uid, "error": "Cannot cancel a completed task"},
            )

        if task_status == "error":
            return JSONResponse(
                status_code=400,
                content={"status": "already_failed", "hash": uid, "error": "Task has already failed"},
            )

        if task_status == "interrupted":
            return JSONResponse(
                status_code=400,
                content={
                    "status": "already_interrupted",
                    "hash": uid,
                    "error": "Task was interrupted by server shutdown; submit a new task to retry",
                },
            )

        if task_status == "cancelled":
            return JSONResponse(
                status_code=400,
                content={"status": "already_cancelled", "hash": uid, "error": "Task has already been cancelled"},
            )

        # Update Redis status first, then remove the queued entry from the stream.
        set_status_in_redis(
            uid,
            {
                "status": "cancelled",
                "hash": uid,
                "message": "Task cancelled by user",
                "worker_id": app_state.worker_id,
            },
            force=True,
        )

        remove_task_from_queue(uid)

        logger.info(f"Task {uid} cancelled by user (was {task_status})")
        return JSONResponse(
            status_code=200,
            content={
                "status": "cancelled",
                "hash": uid,
                "message": f"Task cancelled successfully (was {task_status})",
            },
        )

    except Exception as exc:
        logger.error(f"Failed to cancel task {uid}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "hash": uid, "error": str(exc)},
        )
