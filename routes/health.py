"""FastAPI routes for health and load checks."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from controllers.health import handle_health, handle_load, handle_ready, handle_scaler_depth


# --- Health Router --- #
router = APIRouter(tags=["status"])


@router.get("/health")
async def health() -> JSONResponse:
    """Liveness probe."""
    return await handle_health()


@router.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe."""
    return await handle_ready()


@router.get("/load")
async def load() -> JSONResponse:
    """Current load probe."""
    return await handle_load()


@router.get("/scaler/depth")
async def scaler_depth() -> JSONResponse:
    """Scaler depth probe for KEDA."""
    return await handle_scaler_depth()
    