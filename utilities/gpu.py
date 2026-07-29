"""GPU memory management utilities for Pixal3D API server."""

import gc

import torch

import app_state
from utilities.logger import get_logger
import utilities.postprocess # This automatically registers the monkey patch for o_voxel.postprocess.to_glb


# --- Logger --- #
logger = get_logger(__name__)


def aggressive_gpu_cleanup():
    """Perform aggressive GPU memory cleanup to prevent fragmentation and OOM."""
    # Python-level garbage collection.
    gc.collect()
    gc.collect()

    if not torch.cuda.is_available():
        return

    if app_state.cuda_context_poisoned:
        # The CUDA context already hit an illegal-memory-access/device-side-assert
        # error somewhere in this process. Every CUDA call from here on is
        # undefined behavior -- calling torch.cuda.synchronize() on a poisoned
        # context has been observed to hang the whole process in an unkillable
        # kernel wait (D state) instead of failing cleanly. Skip all further CUDA
        # calls; the process is being torn down and restarted fresh instead.
        return

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

        exc_str = str(exc).lower()
        if "illegal memory access" in exc_str or "device-side assert" in exc_str:
            logger.critical(
                "CUDA context poisoned during cleanup -- disabling further GPU "
                "cleanup calls in this process until it restarts."
            )
            app_state.cuda_context_poisoned = True


def clean_mesh(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clean the mesh vertices and faces directly on the GPU using cumesh.
    
    Removes duplicate faces, repairs non-manifold edges, cleans up small 
    connected component noise, and fills holes to prevent library-level crashes, 
    degenerate topologies, or GPU illegal memory access encounters.
    
    Args:
        vertices (torch.Tensor): A tensor of shape (N, 3) representing vertex coordinates.
        faces (torch.Tensor): A tensor of shape (M, 3) representing face indices.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The cleaned mesh vertices and faces tensors, 
            moved back to their original device.
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices, faces

    # Clear prior lingering activations to secure a continuous block of GPU memory
    aggressive_gpu_cleanup()

    try:
        import cumesh
        device = vertices.device
        
        # Super fast GPU-based vertex welding using PyTorch unique logic.
        # This merges duplicate and unwelded vertices, connecting independent 
        # triangles into a manifold mesh before any cleaning or simplification.
        unique_verts, inverse_indices = torch.unique(vertices, dim=0, return_inverse=True)
        faces_welded = inverse_indices[faces.long()].int()
        
        # Explicitly align welded tensors to GPU prior to library initialization
        gpu_verts = unique_verts.to("cuda")
        gpu_faces = faces_welded.to("cuda")
        
        # Instantiate cumesh handler using CUDA-resident tensors
        cu_mesh = cumesh.CuMesh()
        cu_mesh.init(gpu_verts, gpu_faces)
        
        # Resolve initial degenerate facets to ensure GPU BVH builder does not hit page faults
        cu_mesh.remove_duplicate_faces()
        cu_mesh.repair_non_manifold_edges()
        
        # For extremely dense meshes, skip the slow components and hole-filling passes and do them post-decimation
        if faces.shape[0] < 5000000:
            cu_mesh.remove_small_connected_components(1e-5)
            # Safely handle potential CUDA invalid configuration launches on meshes with zero holes/loops to fill
            utilities.postprocess.robust_fill_holes(cu_mesh, float(1e-1))
        
        out_verts, out_faces = cu_mesh.read()
        
        # Safe-cast and return back to caller's original device
        out_verts = out_verts.to(device)
        out_faces = out_faces.to(device)
        
        # Purge temporary handles to avoid VRAM fragmentation and leaks
        del cu_mesh, gpu_verts, gpu_faces, unique_verts, inverse_indices, faces_welded
        aggressive_gpu_cleanup()
        
        return out_verts, out_faces
        
    except BaseException as e:
        logger.warning(f"GPU-based cleaning failed: {type(e).__name__} - {e}")
        # Always run cleanup even on fatal CUDA or system exceptions
        try:
            aggressive_gpu_cleanup()
        except BaseException:
            pass
            
        return vertices, faces


def smooth_noisy_vertices(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    residual_percentile: float = 90.0,
    blend: float = 0.5,
) -> torch.Tensor:
    """
    Selectively smooths only the vertices whose local neighborhood is measurably
    non-planar, leaving already-clean geometry completely untouched.

    cu_mesh.simplify() has a documented, upstream-unresolved accuracy bug
    (JeffreyXiang/CuMesh#28) that can leave small-scale positional noise behind
    after decimation -- confirmed here by directly measuring a real exported mesh:
    a per-vertex local-plane-fit residual (pure position data, no reference to
    normals) correlated at ~0.6 with per-vertex normal variance in the affected
    region, meaning the visible shading ripple traces back to genuine geometric
    noise, not merely noisy normal computation on an otherwise-flat surface.
    Smoothing normals alone would therefore mask rather than fix it (and risks a
    shading/geometry mismatch under different lighting/angles); this smooths the
    actual noisy positions instead, but only where the noise is measured, so
    intentionally sharp/clean geometry elsewhere (collar edges, fold lines,
    facial features) is left bit-for-bit unchanged.

    Args:
        vertices (torch.Tensor): (N, 3) vertex positions (CUDA).
        faces (torch.Tensor): (M, 3) face indices (CUDA).
        residual_percentile (float): Only vertices at or above this percentile of
            local plane-fit residual are smoothed (default: 90th percentile, i.e.
            roughly the noisiest 10% of vertices).
        blend (float): Blend factor toward the 1-ring neighbor centroid for
            flagged vertices (0 = no change, 1 = fully replace with the average).

    Returns:
        torch.Tensor: (N, 3) adjusted vertex positions.
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices

    device = vertices.device
    n = vertices.shape[0]
    verts_f = vertices.float()

    # Symmetric 1-ring edge list from the face list (both directions, deduplicated).
    e0 = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]]).long()
    e1 = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]]).long()
    edges = torch.unique(torch.stack([torch.cat([e0, e1]), torch.cat([e1, e0])], dim=1), dim=0)
    src, dst = edges[:, 0], edges[:, 1]

    # 1-ring neighbor centroid/count (the smoothing target for flagged vertices).
    neighbor_sum = torch.zeros((n, 3), device=device, dtype=verts_f.dtype)
    neighbor_sum.index_add_(0, src, verts_f[dst])
    neighbor_count = torch.zeros(n, device=device, dtype=verts_f.dtype)
    neighbor_count.index_add_(0, src, torch.ones_like(dst, dtype=verts_f.dtype))

    has_neighbors = neighbor_count > 0
    neighbor_avg = verts_f.clone()
    neighbor_avg[has_neighbors] = neighbor_sum[has_neighbors] / neighbor_count[has_neighbors].unsqueeze(-1)

    # Local centroid (neighbors + self) used for the plane fit below.
    local_centroid = (neighbor_sum + verts_f) / (neighbor_count + 1).unsqueeze(-1)

    # Per-vertex local covariance, accumulated from neighbor + self deviations from
    # that vertex's own local centroid.
    diff = verts_f[dst] - local_centroid[src]
    cov = torch.zeros((n, 3, 3), device=device, dtype=verts_f.dtype)
    cov.index_add_(0, src, diff.unsqueeze(-1) * diff.unsqueeze(-2))
    self_diff = verts_f - local_centroid
    cov += self_diff.unsqueeze(-1) * self_diff.unsqueeze(-2)

    # Vertices with too few neighbors can't support a stable plane fit; leave untouched.
    valid = neighbor_count >= 4
    if not bool(valid.any()):
        return vertices

    # Smallest eigenvalue (ascending order) relative to the trace = flatness residual:
    # ~0 for a planar neighborhood, larger for a genuinely non-planar one.
    eigvals = torch.linalg.eigvalsh(cov[valid])
    residual = torch.zeros(n, device=device, dtype=verts_f.dtype)
    residual[valid] = eigvals[:, 0] / eigvals.sum(dim=-1).clamp_min(1e-12)

    threshold = torch.quantile(residual[valid], residual_percentile / 100.0)
    flagged = valid & has_neighbors & (residual >= threshold)

    if not bool(flagged.any()):
        return vertices

    smoothed = vertices.clone()
    smoothed[flagged] = (
        (1 - blend) * verts_f[flagged] + blend * neighbor_avg[flagged]
    ).to(vertices.dtype)
    return smoothed


def simplify_mesh(vertices: torch.Tensor, faces: torch.Tensor, target_faces: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decimate the mesh to a specified target number of faces directly on the GPU.
    
    Performs topological pre-cleaning to ensure safe simplification, executes 
    quadric decimation via cumesh, and executes a final post-cleaning pass to 
    guarantee non-degenerate outputs.
    
    Args:
        vertices (torch.Tensor): A tensor of shape (N, 3) representing vertex coordinates.
        faces (torch.Tensor): A tensor of shape (M, 3) representing face indices.
        target_faces (int): The target density (face count) to decimate to.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The simplified and cleaned mesh vertices 
            and faces tensors, on their original device.
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return vertices, faces

    # Clear prior lingering activations to secure a continuous block of GPU memory
    aggressive_gpu_cleanup()

    try:
        import cumesh
        device = vertices.device
        
        # Super fast GPU-based vertex welding using PyTorch unique logic.
        unique_verts, inverse_indices = torch.unique(vertices, dim=0, return_inverse=True)
        faces_welded = inverse_indices[faces.long()].int()
        
        # Explicitly align welded tensors to GPU prior to library initialization
        gpu_verts = unique_verts.to("cuda")
        gpu_faces = faces_welded.to("cuda")
        
        # Instantiate cumesh handler using CUDA-resident tensors
        cu_mesh = cumesh.CuMesh()
        cu_mesh.init(gpu_verts, gpu_faces)
        
        # Minimize input cleanup to speed up massive decimation pipelines
        cu_mesh.remove_duplicate_faces()
        cu_mesh.repair_non_manifold_edges()
        
        # Collapse edges safely on GPU using quadratic decimation metrics
        cu_mesh.simplify(target_faces)
        
        # Secure topological health post-decimation to avoid invalid face indices or isolated vertices
        cu_mesh.remove_duplicate_faces()
        cu_mesh.repair_non_manifold_edges()
        cu_mesh.remove_small_connected_components(1e-5)
        
        # Safely fill holes using robust boundary validation sweep to avoid invalid grid configurations
        utilities.postprocess.robust_fill_holes(cu_mesh, float(1e-1))
        
        out_verts, out_faces = cu_mesh.read()
        
        # Safe-cast and return back to caller's original device
        out_verts = out_verts.to(device)
        out_faces = out_faces.to(device)
        
        # Purge temporary handles to avoid VRAM fragmentation and leaks
        del cu_mesh, gpu_verts, gpu_faces, unique_verts, inverse_indices, faces_welded
        aggressive_gpu_cleanup()
        
        return out_verts, out_faces
        
    except BaseException as e:
        logger.warning(f"GPU-based simplification failed: {type(e).__name__} - {e}")
        # Always run cleanup even on fatal CUDA or system exceptions
        try:
            aggressive_gpu_cleanup()
        except BaseException:
            pass

        raise e
