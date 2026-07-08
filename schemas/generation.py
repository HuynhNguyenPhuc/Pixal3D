"""Pydantic models for Pixal3D API server."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request model for 3D generation API."""

    hash: str = Field(
        ...,
        description="The hash of the 3D generation task (required for uniqueness)",
        min_length=1,
    )
    image: str = Field(
        ...,
        description="Canonical GCS URL or HTTP(S) URL for the input image.",
        example="gs://my-input-bucket/sticker-inputs/4f9536f5/thumbnail.png",
    )
    content_type: str = Field(
        ...,
        description="Content type of the input image (e.g. image/png, image/jpeg)",
    )
    seed: int = Field(42, description="Random seed for reproducible generation", ge=0, le=2**32 - 1)
    decimation_target: int = Field(
        500000,
        description="Target face count for mesh decimation",
        ge=100000,
        le=1000000,
    )
    texture_size: int = Field(
        2048,
        description="Resolution of the output texture",
        ge=1024,
        le=4096,
    )
    ss_guidance_strength: float = Field(7.5, description="Guidance strength for sparse structure sampling", ge=1.0, le=10.0)
    ss_guidance_rescale: float = Field(0.7, description="Guidance rescale for sparse structure sampling", ge=0.0, le=1.0)
    ss_sampling_steps: int = Field(12, description="Number of sampling steps for sparse structure", ge=1, le=50)
    ss_rescale_t: float = Field(5.0, description="Rescale T for sparse structure sampling", ge=1.0, le=6.0)
    shape_slat_guidance_strength: float = Field(7.5, description="Guidance strength for shape SLat sampling", ge=1.0, le=10.0)
    shape_slat_guidance_rescale: float = Field(0.5, description="Guidance rescale for shape SLat sampling", ge=0.0, le=1.0)
    shape_slat_sampling_steps: int = Field(12, description="Number of sampling steps for shape SLat", ge=1, le=50)
    shape_slat_rescale_t: float = Field(3.0, description="Rescale T for shape SLat sampling", ge=1.0, le=6.0)
    tex_slat_guidance_strength: float = Field(1.0, description="Guidance strength for texture SLat sampling", ge=1.0, le=10.0)
    tex_slat_guidance_rescale: float = Field(0.0, description="Guidance rescale for texture SLat sampling", ge=0.0, le=1.0)
    tex_slat_sampling_steps: int = Field(12, description="Number of sampling steps for texture SLat", ge=1, le=50)
    tex_slat_rescale_t: float = Field(3.0, description="Rescale T for texture SLat sampling", ge=1.0, le=6.0)
    fov: float = Field(-1.0, description="Manual camera FOV in radians. Set to -1.0 for automatic estimation.", ge=-1.0, le=3.0)
    no_webp: bool = Field(True, description="Disable WebP textures in output GLB")


class GenerationResponse(BaseModel):
    """Response model for generation endpoints."""

    status: str = Field(..., description="Task status (completed or queued)")
    hash: str = Field(..., description="Task hash identifier")
    filename: Optional[str] = Field(None, description="Generated model filename")
    url: Optional[str] = Field(None, description="Canonical GCS URL for the generated model")


class StatusResponse(BaseModel):
    """Response model for status endpoint."""

    status: str = Field(..., description="Status of the generation task")
    hash: Optional[str] = Field(None, description="Task hash identifier")
    filename: Optional[str] = Field(None, description="Generated model filename (completed only)")
    url: Optional[str] = Field(None, description="Canonical GCS URL for the generated model (completed only)")
    message: Optional[str] = Field(None, description="Additional message or error description")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Health status")
    worker_id: str = Field(..., description="Worker identifier")
