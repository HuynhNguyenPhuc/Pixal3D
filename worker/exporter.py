"""GLB mesh export with multi-stage GPU-safety fallbacks.

Runs cumesh/o_voxel GLB postprocessing in-process. Topology cleanup, proactive
decimation, and a multi-stage fallback chain (remesh -> no remesh -> GPU-decimated
+ lower texture) let CUDA OOMs during export degrade gracefully instead of failing
the task outright.
"""

import numpy as np
import torch

import config
import utilities.postprocess  # Register monkey-patch for o_voxel.postprocess.to_glb
import o_voxel.postprocess
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
        RuntimeError: If the mesh exceeds the hard face ceiling, or export fails
            for a reason other than CUDA OOM.
        torch.OutOfMemoryError: If a CUDA OOM persists through all fallback attempts.
    """
    logger.info(f"Exporting GLB: {vertices.shape[0]:,} vertices, {faces.shape[0]:,} faces.")

    # --- Step 1: Clean Topology --- #
    vertices, faces = clean_mesh(vertices, faces)

    # --- Step 2: Proactive GPU Simplification --- #
    simplify_trigger = min(config.SIMPLIFICATION_THRESHOLD_FACES, config.MESH_HARD_FACE_CEILING)

    if faces.shape[0] >= simplify_trigger:
        logger.info(f"Mesh faces ({faces.shape[0]:,}) exceed threshold ({simplify_trigger:,}). Simplifying...")
        simplified = False

        for attempt in range(2):
            try:
                vertices, faces = simplify_mesh(vertices, faces, config.SIMPLIFICATION_TARGET_FACES)
                simplified = True
                break
            except BaseException as e:
                logger.warning(f"GPU Simplification attempt {attempt + 1}/2 failed: {e}")
                aggressive_gpu_cleanup()

        if not simplified:
            logger.warning("GPU Simplification failed after retry. Proceeding with cleaned mesh.")

    # --- Step 3: Hard Ceiling Safeguard Check --- #
    if faces.shape[0] > config.MESH_HARD_FACE_CEILING:
        error_msg = f"Mesh face count ({faces.shape[0]:,}) exceeds hard ceiling ({config.MESH_HARD_FACE_CEILING:,})."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # --- Step 4: Primary Export Attempt (remesh=True) --- #
    try:
        mesh_result = o_voxel.postprocess.to_glb(
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

    # --- Step 5: Fallback Attempt 1 (remesh=False) --- #
    try:
        mesh_result = o_voxel.postprocess.to_glb(
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

    # --- Step 6: Final Fallback Attempt (GPU Decimation + Lower Texture) --- #
    gpu_target = max(decimation_target, 200000)

    try:
        vertices_fallback, faces_fallback = simplify_mesh(vertices, faces, gpu_target)
    except BaseException:
        vertices_fallback, faces_fallback = vertices, faces

    mesh_result = o_voxel.postprocess.to_glb(
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
