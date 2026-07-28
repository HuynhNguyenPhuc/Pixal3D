"""GLB mesh export with multi-stage GPU-safety fallbacks."""

import numpy as np
import torch

from utilities.postprocess import robust_to_glb
from utilities.error_handling import is_cuda_out_of_memory
from utilities.gpu import aggressive_gpu_cleanup, clean_mesh, simplify_mesh
from utilities.logger import get_logger


# --- Logger --- #
logger = get_logger(__name__)

# Y/Z axis swap rotation matrix to convert from internal coordinate system to GLB standard.
_EXPORT_ROTATION = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def _finalize_glb_bytes(mesh, no_webp: bool) -> bytes:
    """Finalizes the GLB bytes by applying the export rotation and exporting to GLB format.

    Args:
        mesh: The Trimesh object produced by o_voxel postprocessing.
        no_webp (bool): If True, disables WEBP texture compression in the GLB export.

    Returns:
        bytes: The exported GLB binary data.
    """
    mesh.apply_transform(_EXPORT_ROTATION)
    return mesh.export(file_type="glb", extension_webp=not no_webp)


def export_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attrs: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: list,
    res: int,
    decimation_target: int,
    texture_size: int,
    no_webp: bool = False,
) -> bytes:
    """Exports a GLB from mesh geometry and PBR voxel attributes.

    Args:
        vertices (torch.Tensor): Mesh vertices tensor (GPU).
        faces (torch.Tensor): Mesh faces tensor (GPU).
        attrs (torch.Tensor): Mesh attributes volume tensor (GPU).
        coords (torch.Tensor): Mesh coordinates tensor (GPU).
        attr_layout (list): Attribute layout definition.
        res (int): Voxel resolution.
        decimation_target (int): Target face count for decimation.
        texture_size (int): Export texture resolution.
        no_webp (bool): If True, disables WEBP texture compression in the GLB export.

    Returns:
        bytes: The exported GLB binary data, rotated to standard orientation.

    Raises:
        RuntimeError: If export fails for a reason other than CUDA OOM.
        torch.OutOfMemoryError: If a CUDA OOM persists through all fallback attempts.
    """
    logger.info(f"Exporting GLB: {vertices.shape[0]:,} vertices, {faces.shape[0]:,} faces.")

    # --- Step 1: Clean Topology --- #
    vertices, faces = clean_mesh(vertices, faces)

    # --- Step 2: Primary Export Attempt (remesh=True) --- #
    try:
        mesh_result = robust_to_glb(
            vertices=vertices,
            faces=faces,
            attr_volume=attrs,
            coords=coords,
            attr_layout=attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )
        return _finalize_glb_bytes(mesh_result, no_webp)

    except Exception as exc:
        if not is_cuda_out_of_memory(exc):
            raise

        logger.warning(f"OOM during primary GLB export: {exc}. Retrying without remesh...")
        aggressive_gpu_cleanup()

    # --- Step 3: Fallback to remesh=False --- #
    try:
        mesh_result = robust_to_glb(
            vertices=vertices,
            faces=faces,
            attr_volume=attrs,
            coords=coords,
            attr_layout=attr_layout,
            grid_size=res,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=False,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=False,
        )
        return _finalize_glb_bytes(mesh_result, no_webp)

    except Exception as exc_fallback:
        if not is_cuda_out_of_memory(exc_fallback):
            raise

        logger.warning(f"OOM during fallback GLB export: {exc_fallback}. Lowering texture resolution...")
        aggressive_gpu_cleanup()

    # --- Step 4: Final Fallback Attempt (GPU Decimation + Lower Texture) --- #
    gpu_target = max(decimation_target, 100000)

    try:
        vertices_fallback, faces_fallback = simplify_mesh(vertices, faces, gpu_target)
    except BaseException:
        vertices_fallback, faces_fallback = vertices, faces

    mesh_result = robust_to_glb(
        vertices=vertices_fallback,
        faces=faces_fallback,
        attr_volume=attrs,
        coords=coords,
        attr_layout=attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=min(texture_size, 1024),
        remesh=False,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=False,
    )

    return _finalize_glb_bytes(mesh_result, no_webp)
