"""GPU memory management utilities for Pixal3D API server."""

import gc

import torch

from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)


def aggressive_gpu_cleanup():
    """Perform aggressive GPU memory cleanup to prevent fragmentation and OOM."""
    # Python-level garbage collection.
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        try:
            # Synchronize to ensure all operations complete.
            torch.cuda.synchronize()

            # Clear the CUDA allocator cache.
            torch.cuda.empty_cache()

            # Collect IPC handles that are no longer needed.
            torch.cuda.ipc_collect()

            # Synchronize again after cleanup.
            torch.cuda.synchronize()

            # Reset peak memory stats for fresh tracking.
            torch.cuda.reset_peak_memory_stats()

        except Exception as exc:
            logger.warning(f"GPU cleanup error: {exc}")
