"""Error classification helpers for task execution."""

import torch


def is_cuda_out_of_memory(exc: BaseException) -> bool:
    """
    True only for genuinely recoverable CUDA/CuMesh out-of-memory failures.

    Illegal memory access and device-side assert errors mean the CUDA context
    itself is corrupted -- every subsequent CUDA call in the process is
    undefined behavior until the process is torn down. They must never be
    treated as OOM (cleanup-and-retry cannot recover from them), even though
    their message text also contains "cuda error".

    Args:
        exc (BaseException): The failure raised during a CUDA/CuMesh operation.

    Returns:
        bool: True if this is a recoverable CUDA/CuMesh out-of-memory error.
    """
    normalized = str(exc).lower()

    if "illegal memory access" in normalized or "device-side assert" in normalized:
        return False

    return (
        isinstance(exc, torch.OutOfMemoryError)
        or "cuda out of memory" in normalized
        or ("cuda error" in normalized and "out of memory" in normalized)
        or ("cumesh" in normalized and "out of memory" in normalized)
        or ("error code: 2" in normalized and "out of memory" in normalized)
    )


def classify_task_error(exc: Exception, params: dict | None = None) -> dict:
    """
    Classify execution failures into retry-friendly API error payloads.

    Args:
        exc (Exception): The failure raised during task execution.
        params (dict | None): Reserved for future parameter-aware classification.

    Returns:
        dict: Structured error payload used by runtime status writes.
    """
    del params

    message = str(exc)

    if is_cuda_out_of_memory(exc):
        return {
            "message": message,
            "error_code": "CUDA_OOM",
            "error_type": "resource_exhausted",
            "retriable": True,
        }

    # Fall back to a generic internal failure when the error does not match a
    # known retryable resource issue.
    return {
        "message": message,
        "error_code": "GENERATION_FAILED",
        "error_type": "internal",
        "retriable": False,
    }
