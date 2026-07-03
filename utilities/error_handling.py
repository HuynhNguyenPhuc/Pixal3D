"""Error classification helpers for task execution."""


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
    normalized = message.lower()

    # CUDA and CuMesh tend to surface OOM failures through slightly different
    # strings, so keep the matcher broad and operational rather than exact.
    is_cuda_oom = (
        ("cuda error" in normalized and "out of memory" in normalized)
        or ("cumesh" in normalized and "out of memory" in normalized)
        or ("error code: 2" in normalized and "out of memory" in normalized)
        or ("cuda out of memory" in normalized)
    )

    if is_cuda_oom:
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
