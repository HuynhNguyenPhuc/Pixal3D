"""Isolated GLB mesh export module using multiprocessing.

Runs cumesh/o_voxel GLB postprocessing inside a dedicated spawned child process.
If cumesh or nvdiffrast triggers a low-level C++/CUDA error (illegal memory access,
segfault, device-side assert), ONLY the isolated child process terminates.
The main worker process retains its loaded ML models in VRAM without needing a
time-consuming ~3.5-minute restart.
"""

import multiprocessing as mp
import numpy as np
import torch

import config
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


def _export_process_entry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attrs: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: list,
    res: int,
    decimation_target: int,
    texture_size: int,
    no_webp: bool,
    result_queue: mp.Queue,
) -> None:
    """Entry point for the isolated GLB export process.

    Runs in a spawned child process with a fresh CUDA context. Executes topology
    cleanup, proactive decimation, and multi-stage fallback GLB generation.

    Args:
        vertices (torch.Tensor): CPU Mesh vertices tensor.
        faces (torch.Tensor): CPU Mesh faces tensor.
        attrs (torch.Tensor): CPU Mesh attributes volume tensor.
        coords (torch.Tensor): CPU Mesh coordinates tensor.
        attr_layout (list): Attribute layout definition.
        res (int): Voxel resolution.
        decimation_target (int): Target face count for decimation.
        texture_size (int): Export texture resolution.
        no_webp (bool): If True, disables WEBP texture compression.
        result_queue (mp.Queue): Multiprocessing queue to return result or error payload.
    """
    try:
        import utilities.postprocess  # Register monkey-patch for o_voxel.postprocess.to_glb
        import o_voxel.postprocess

        # --- Device Setup --- #
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Child Exporter Process started on device: {device}")

        mesh_vertices = vertices.to(device)
        mesh_faces = faces.to(device)
        mesh_attrs = attrs.to(device)
        mesh_coords = coords.to(device)

        logger.info(f"Input mesh loaded to GPU: {mesh_vertices.shape[0]:,} vertices, {mesh_faces.shape[0]:,} faces.")

        # --- Step 1: Clean Topology --- #
        mesh_vertices, mesh_faces = clean_mesh(mesh_vertices, mesh_faces)

        # --- Step 2: Proactive GPU Simplification --- #
        simplify_trigger = min(config.SIMPLIFICATION_THRESHOLD_FACES, config.MESH_HARD_FACE_CEILING)

        if mesh_faces.shape[0] >= simplify_trigger:
            logger.info(f"Mesh faces ({mesh_faces.shape[0]:,}) exceed threshold ({simplify_trigger:,}). Simplifying...")
            simplified = False

            for attempt in range(2):
                try:
                    mesh_vertices, mesh_faces = simplify_mesh(
                        mesh_vertices, mesh_faces, config.SIMPLIFICATION_TARGET_FACES
                    )
                    simplified = True
                    break
                except BaseException as e:
                    logger.warning(f"GPU Simplification attempt {attempt + 1}/2 failed: {e}")
                    aggressive_gpu_cleanup()

            if not simplified:
                logger.warning("GPU Simplification failed after retry. Proceeding with cleaned mesh.")

        # --- Step 3: Hard Ceiling Safeguard Check --- #
        if mesh_faces.shape[0] > config.MESH_HARD_FACE_CEILING:
            error_msg = f"Mesh face count ({mesh_faces.shape[0]:,}) exceeds hard ceiling ({config.MESH_HARD_FACE_CEILING:,})."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # --- Step 4: Primary Export Attempt (remesh=True) --- #
        try:
            mesh_result = o_voxel.postprocess.to_glb(
                vertices=mesh_vertices,
                faces=mesh_faces,
                attr_volume=mesh_attrs,
                coords=mesh_coords,
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
            glb_bytes = _finalize_glb_bytes(mesh_result, no_webp)
            result_queue.put(("SUCCESS", glb_bytes))
            return

        except Exception as exc:
            if not is_cuda_out_of_memory(exc):
                raise exc

            logger.warning(f"OOM during primary GLB export: {exc}. Retrying without remesh...")
            aggressive_gpu_cleanup()

        # --- Step 5: Fallback Attempt 1 (remesh=False) --- #
        try:
            mesh_result = o_voxel.postprocess.to_glb(
                vertices=mesh_vertices,
                faces=mesh_faces,
                attr_volume=mesh_attrs,
                coords=mesh_coords,
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
            glb_bytes = _finalize_glb_bytes(mesh_result, no_webp)
            result_queue.put(("SUCCESS", glb_bytes))
            return

        except Exception as exc_fallback:
            if not is_cuda_out_of_memory(exc_fallback):
                raise exc_fallback

            logger.warning(f"OOM during fallback GLB export: {exc_fallback}. Lowering texture resolution...")
            aggressive_gpu_cleanup()

        # --- Step 6: Final Fallback Attempt (GPU Decimation + Lower Texture) --- #
        gpu_target = max(decimation_target, 200000)

        try:
            mesh_vertices_fallback, mesh_faces_fallback = simplify_mesh(mesh_vertices, mesh_faces, gpu_target)
        except BaseException as e_gpu:
            mesh_vertices_fallback, mesh_faces_fallback = mesh_vertices, mesh_faces

        mesh_result = o_voxel.postprocess.to_glb(
            vertices=mesh_vertices_fallback,
            faces=mesh_faces_fallback,
            attr_volume=mesh_attrs,
            coords=mesh_coords,
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
        glb_bytes = _finalize_glb_bytes(mesh_result, no_webp)
        result_queue.put(("SUCCESS", glb_bytes))
        return

    except Exception as e:
        logger.error(f"Isolated GLB export process failed: {e}")
        result_queue.put(("ERROR", str(e), is_cuda_out_of_memory(e)))


def export_glb_isolated(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attrs: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: list,
    res: int,
    decimation_target: int,
    texture_size: int,
    no_webp: bool = False,
    timeout_seconds: int = 300,
) -> bytes:
    """Exports GLB in an isolated spawned subprocess.

    Creates a fresh child process via multiprocessing (spawn method) to isolate CUDA/CuMesh
    export operations. If the child process crashes due to low-level CUDA errors, the main
    worker process remains untouched.

    Args:
        vertices (torch.Tensor): Mesh vertices tensor.
        faces (torch.Tensor): Mesh faces tensor.
        attrs (torch.Tensor): Mesh attributes volume tensor.
        coords (torch.Tensor): Mesh coordinates tensor.
        attr_layout (list): Attribute layout definition.
        res (int): Voxel resolution.
        decimation_target (int): Target face count for decimation.
        texture_size (int): Export texture resolution.
        no_webp (bool): If True, disables WEBP texture compression in the GLB export.
        timeout_seconds (int): Maximum time in seconds to wait for child process.

    Returns:
        bytes: The exported GLB binary data, rotated to standard orientation.

    Raises:
        TimeoutError: If the child process exceeds timeout_seconds.
        RuntimeError: If export fails or child process crashes.
        torch.OutOfMemoryError: If a CUDA Out of Memory error occurs during export.
    """
    logger.info(f"Spawning isolated export process (timeout={timeout_seconds}s)...")

    # --- Transfer Tensors to CPU for IPC --- #
    vertices_cpu = vertices.detach().cpu()
    faces_cpu = faces.detach().cpu()
    attrs_cpu = attrs.detach().cpu()
    coords_cpu = coords.detach().cpu()

    # --- Spawn Child Process --- #
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(
        target=_export_process_entry,
        args=(
            vertices_cpu,
            faces_cpu,
            attrs_cpu,
            coords_cpu,
            attr_layout,
            res,
            decimation_target,
            texture_size,
            no_webp,
            result_queue,
        ),
    )

    p.start()
    logger.info(f"Child export process started with PID: {p.pid}")

    p.join(timeout=timeout_seconds)

    # --- Handle Timeout --- #
    if p.is_alive():
        logger.error(f"Isolated export child process (PID {p.pid}) timed out after {timeout_seconds}s. Terminating process.")
        p.terminate()
        p.join(timeout=5)
        raise TimeoutError(f"GLB export process timed out after {timeout_seconds}s.")

    # --- Handle Abnormal Process Exit --- #
    exit_code = p.exitcode
    if exit_code != 0:
        logger.error(f"Isolated export child process (PID {p.pid}) crashed with exit code {exit_code}.")
        raise RuntimeError(f"GLB export process crashed with exit code {exit_code} (likely low-level CUDA/CuMesh error).")

    # --- Handle Empty Output Queue --- #
    if result_queue.empty():
        logger.error(f"Child export process (PID {p.pid}) exited with code 0 but returned no result.")
        raise RuntimeError("GLB export child process exited without returning a result.")

    # --- Parse Result Payload --- #
    res_tuple = result_queue.get()
    status = res_tuple[0]

    if status == "SUCCESS":
        logger.info(f"Isolated export process (PID {p.pid}) completed successfully. Received {len(res_tuple[1]):,} bytes.")
        return res_tuple[1]
    else:
        err_msg = res_tuple[1]
        is_oom = res_tuple[2]
        logger.error(f"Isolated export process reported an error: {err_msg} (is_oom={is_oom})")

        if is_oom:
            raise torch.OutOfMemoryError(err_msg)
        else:
            raise RuntimeError(err_msg)

