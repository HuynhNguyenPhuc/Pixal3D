"""FastAPI routes for generation tasks."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from controllers.generation import handle_cancel, handle_send, handle_status
from schemas.generation import GenerationRequest


# --- Generation Router --- #
router = APIRouter(prefix="/tasks", tags=["Generation"])


@router.post("/submit")
async def submit(request_data: GenerationRequest) -> JSONResponse:
    """Submit a new generation task."""
    return await handle_send(request_data)


@router.get("/status/{uid}")
async def status(uid: str) -> JSONResponse:
    """Get the current task status."""
    return await handle_status(uid)


@router.post("/cancel/{uid}")
async def cancel(uid: str) -> JSONResponse:
    """Cancel a queued or running task."""
    return await handle_cancel(uid)
