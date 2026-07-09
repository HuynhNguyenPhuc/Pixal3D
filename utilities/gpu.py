"""GPU memory management utilities for Pixal3D API server."""

import gc

import torch

from utilities.logger import get_logger
import utilities.postprocess # This automatically registers the monkey patch for o_voxel.postprocess.to_glb


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
            try:
                cu_mesh.fill_holes(float(3e-2))
            except RuntimeError as e:
                if "invalid configuration argument" not in str(e):
                    raise e
        
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
        tuple[torch.Tensor, torch.Tensor]: The simplified and cleaned mesh mesh vertices 
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
        
        # Minimize input cleanup to speed up massive decimation pipelines (mesh is already clean_mesh)
        cu_mesh.remove_duplicate_faces()
        cu_mesh.repair_non_manifold_edges()
        
        # Collapse edges safely on GPU using quadratic decimation metrics
        cu_mesh.simplify(target_faces)
        
        # Secure topological health post-decimation to avoid invalid face indices or isolated vertices
        cu_mesh.remove_duplicate_faces()
        cu_mesh.repair_non_manifold_edges()
        cu_mesh.remove_small_connected_components(1e-5)
        # Safely handle potential CUDA invalid configuration launches on meshes with zero holes/loops to fill
        try:
            cu_mesh.fill_holes(float(3e-2))
        except RuntimeError as e:
            if "invalid configuration argument" not in str(e):
                raise e
        
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

