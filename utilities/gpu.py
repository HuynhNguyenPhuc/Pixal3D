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


def clean_mesh_vertices_faces(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clean mesh vertices and faces to prevent cumesh illegal memory access.
    Merges duplicate vertices, removes degenerate and duplicate faces,
    and removes unreferenced vertices using trimesh.
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices, faces

    try:
        import trimesh
        # Move to CPU numpy for trimesh processing
        device = vertices.device
        v_np = vertices.detach().cpu().numpy()
        f_np = faces.detach().cpu().numpy()
        
        # Create trimesh object with automatic processing/merging/cleaning
        t_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=True)
        
        # Perform explicit additional cleanup steps to be absolutely sure
        t_mesh.update_faces(t_mesh.nondegenerate_faces() & t_mesh.unique_faces())
        t_mesh.remove_unreferenced_vertices()
        
        # Convert back to torch Tensors on the original device
        clean_v = torch.from_numpy(t_mesh.vertices).float().to(device)
        clean_f = torch.from_numpy(t_mesh.faces).int().to(device)
        return clean_v, clean_f
    except Exception as e:
        logger.warning(f"Failed to clean mesh with trimesh: {e}")
        return vertices, faces

