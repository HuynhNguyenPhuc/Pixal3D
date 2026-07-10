import gc
import math
from typing import Dict, Union, List

import cv2
import numpy as np
import trimesh
import trimesh.visual
from PIL import Image

import torch

import cumesh
import nvdiffrast.torch as dr
import o_voxel
from flex_gemm.ops.grid_sample import grid_sample_3d


def robust_to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Highly stable, crash-free drop-in replacement for o_voxel.postprocess.to_glb.
    
    This function processes raw 3D mesh vertices, faces, and sparse voxel volume 
    features to produce a fully textured, PBR-material-mapped GLB file.
    
    It prevents crashes (like Out-Of-Memory or CUDA illegal memory access errors)
    by running isolated try-catch stages, welding vertices first, and bypassing
    problematic topological repair operations on Dual-Contoured surfaces.
    """
    from tqdm import tqdm
    
    # Clean up GPU memory immediately to maximize contiguous free VRAM
    def local_cleanup():
        try:
            from utilities.gpu import aggressive_gpu_cleanup
            aggressive_gpu_cleanup()
        except BaseException:
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except BaseException:
                    pass

    # ------------------------------------------------------------------------
    # Step 1: Input Tensor Validation & Normalization
    # ------------------------------------------------------------------------
    # Ensure bounding boxes, voxel dimensions, and grid dimensions are in the correct float/integer tensor formats on the GPU.
    try:
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
        assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
        assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
        assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
        assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

        # Find either voxel size or grid size depending on what is provided
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = [voxel_size, voxel_size, voxel_size]
            if isinstance(voxel_size, (list, tuple)):
                voxel_size = np.array(voxel_size)
            if isinstance(voxel_size, np.ndarray):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            assert grid_size is not None, "Either voxel_size or grid_size must be provided"
            if isinstance(grid_size, int):
                grid_size = [grid_size, grid_size, grid_size]
            if isinstance(grid_size, (list, tuple)):
                grid_size = np.array(grid_size)
            if isinstance(grid_size, np.ndarray):
                grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size
        
        # Explicitly align tracking tensors to coords device (GPU) to prevent multi-device assertion errors
        voxel_size = voxel_size.to(coords.device)
        grid_size = grid_size.to(coords.device)
        
        assert isinstance(voxel_size, torch.Tensor)
        assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
        assert isinstance(grid_size, torch.Tensor)
        assert grid_size.dim() == 1 and grid_size.size(0) == 3

        # Force remesh to False if resolution is extremely high to prevent CUDA OOM inside CuMesh
        if remesh and grid_size is not None:
            max_res = int(grid_size.max().item())
            if max_res > 512:
                print(f"[Safeguard] Resolution {max_res} is too high for Dual Contouring remeshing. Forcing remesh=False to prevent GPU OOM.")
                remesh = False
        
    except BaseException as norm_err:
        print(f"[Fatal] to_glb input validation failed: {norm_err}")
        local_cleanup()
        raise norm_err

    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # ------------------------------------------------------------------------
    # Step 2: Weld Duplicate Vertices
    # ------------------------------------------------------------------------
    # Merges vertices sharing identical coordinates into a single unified vertex.
    # This connects isolated triangles into a manifold 3D surface, which stops
    # later decimation passes from collapsing edges into illegal geometry and 
    # causing CuMesh/CUDA page faults.
    try:
        unique_verts, inverse_indices = torch.unique(vertices, dim=0, return_inverse=True)
        faces_welded = inverse_indices[faces.long()].int()
        
        vertices = unique_verts.cuda()
        faces = faces_welded.cuda()
        
    except BaseException as dev_err:
        print(f"[Fatal] to_glb failed aligning tensors to current CUDA device: {dev_err}")
        local_cleanup()
        raise dev_err

    # ------------------------------------------------------------------------
    # Step 3: Mesh Handler Construction & Boundary/Hole Healing
    # ------------------------------------------------------------------------
    # Initialize CuMesh and heal visual gaps/open boundaries using a small perimeter limit.
    # We gracefully ignore CUDA driver-launch failures on meshes with 0 holes.
    try:
        mesh = cumesh.CuMesh()
        mesh.init(vertices, faces)
        
        try:
            mesh.fill_holes(float(3e-2))
        except RuntimeError as e:
            if "invalid configuration argument" not in str(e):
                raise e
                
        if verbose:
            print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        vertices, faces = mesh.read()
        
    except BaseException as cumesh_init_err:
        print(f"[Fatal] cumesh.CuMesh instantiation / validation failed: {cumesh_init_err}")
        local_cleanup()
        raise cumesh_init_err

    if use_tqdm:
        pbar.update(1)

    # ------------------------------------------------------------------------
    # Step 4: Spatial Index Tree (BVH) Instantiation
    # ------------------------------------------------------------------------
    # Construct a GPU-accelerated cuBVH search tree. This tree lets us map 
    # simplified low-resolution vertices back to their exact high-resolution
    # counterparts, preserving precise texture mapping color features.
    try:
        if use_tqdm:
            pbar.set_description("Building BVH")
        if verbose:
            print(f"Building BVH for current mesh...", end='', flush=True)
            
        bvh = cumesh.cuBVH(vertices, faces)
        
        if verbose:
            print("Done")
            
    except BaseException as bvh_err:
        print(f"[Fatal] Spatial cuBVH tree construction failed on vertices/faces: {bvh_err}")
        local_cleanup()
        raise bvh_err

    if use_tqdm:
        pbar.update(1)

    # ------------------------------------------------------------------------
    # Step 5: Topological Processing & Quadric Simplification (Decimation)
    # ------------------------------------------------------------------------
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")

    try:
        # --- BRANCH B: Remeshing Pipeline (Narrow-Band Dual Contouring) ---
        if remesh:
            try:
                center = aabb.mean(dim=0)
                scale = (aabb[1] - aabb[0]).max().item()
                resolution = grid_size.max().item()
                
                # Reconstruct the 3D surface envelope using narrow-band Dual Contouring (Voxelization)
                mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
                    vertices, faces,
                    center = center,
                    scale = (resolution + 3 * remesh_band) / resolution * scale,
                    resolution = resolution,
                    band = remesh_band,
                    project_back = remesh_project, 
                    verbose = verbose,
                    bvh = bvh,
                ))
                if verbose:
                    print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                
                # Decimate the reconstructed remeshed surface down to target density
                mesh.simplify(decimation_target, verbose=verbose)
                
                if verbose:
                    print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            except BaseException as remesh_err:
                print(f"[Warning] Dual Contouring remeshing failed: {remesh_err}. Falling back to standard non-remeshed simplification Branch A.")
                local_cleanup()
                remesh = False
                # Re-initialize mesh with the original welded vertices and faces
                mesh = cumesh.CuMesh()
                mesh.init(vertices, faces)

        # --- BRANCH A: Standard Simplification (No dual contouring remesh) ---
        if not remesh:
            # Clean duplicate triangles before collapsing edges
            mesh.remove_duplicate_faces()
            
            # Step A.1: Preliminary edge collapse targeting 3x the target count
            mesh.simplify(decimation_target * 3, verbose=verbose)
            if verbose:
                print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
            # Step A.2: Topological healing sweeps (remove non-manifolds and fill holes)
            mesh.remove_duplicate_faces()
            mesh.remove_small_connected_components(1e-5)
            try:
                mesh.fill_holes(float(3e-2))
            except RuntimeError as e:
                if "invalid configuration argument" not in str(e):
                    raise e
            if verbose:
                print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                
            # Step A.3: Precise final decimation targeting requested target density
            mesh.simplify(decimation_target, verbose=verbose)
            if verbose:
                print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
            # Step A.4: Post-simplification validation sweeps
            mesh.remove_duplicate_faces()
            mesh.remove_small_connected_components(1e-5)
            try:
                mesh.fill_holes(float(3e-2))
            except RuntimeError as e:
                if "invalid configuration argument" not in str(e):
                    raise e
            if verbose:
                print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
                
            mesh.unify_face_orientations()
                
    except BaseException as topo_err:
        print(f"[Fatal] Topological processing or simplification failed: {topo_err}")
        local_cleanup()
        raise topo_err

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # ------------------------------------------------------------------------
    # Step 6: UV Atlas Chart Generation & Parameterization (uv_unwrap)
    # ------------------------------------------------------------------------
    # Flatten the 3D surface onto a 2D plane (UV space) to create layout charts.
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")

    try:
        # Convert floating cluster arguments to strict types to prevent configuration faults in xatlas/cumesh.
        cone_angle = float(mesh_cluster_threshold_cone_half_angle_rad) if mesh_cluster_threshold_cone_half_angle_rad is not None else float(np.radians(90.0))
        refine_iters = int(mesh_cluster_refine_iterations) if mesh_cluster_refine_iterations is not None else 0
        global_iters = int(mesh_cluster_global_iterations) if mesh_cluster_global_iterations is not None else 1
        smooth_str = int(mesh_cluster_smooth_strength) if mesh_cluster_smooth_strength is not None else 1

        # Fallback to direct xatlas parameterization if fast charting fails/is unstable.
        try:
            out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
                compute_charts_kwargs={
                    "threshold_cone_half_angle_rad": cone_angle,
                    "refine_iterations": refine_iters,
                    "global_iterations": global_iters,
                    "smooth_strength": smooth_str,
                    "area_penalty_weight": 0.1,
                    "perimeter_area_ratio_weight": 0.0001,
                },
                return_vmaps=True,
                verbose=verbose,
            )
            
        except BaseException as atlas_inner_err:
            print(f"[Warning] Fast charting failed: {atlas_inner_err}. Retrying with absolute zero weights...")
            try:
                out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
                    compute_charts_kwargs={
                        "threshold_cone_half_angle_rad": float(np.radians(90.0)),
                        "refine_iterations": 0,
                        "global_iterations": 1,
                        "smooth_strength": 0,
                        "area_penalty_weight": 0.0,
                        "perimeter_area_ratio_weight": 0.0,
                    },
                    return_vmaps=True,
                    verbose=verbose,
                )
            except BaseException as default_fallback_err:
                print(f"[Warning] Zero weight mapping failed: {default_fallback_err}. Retrying with direct default unwrapping...")
                out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
                    return_vmaps=True,
                    verbose=verbose,
                )
                
    except BaseException as unwrap_err:
        print(f"[Warning] Primary UV unwrap failed: {unwrap_err}. Running extremely aggressive topological healing fallback...")
        try:
            # If unwrapping crashes due to lingering complex/non-manifold edges, heal the topology aggressively.
            local_cleanup()
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            mesh.remove_small_connected_components(1e-4)
            try:
                mesh.fill_holes(max_hole_perimeter=0.1)
            except RuntimeError as e:
                if "invalid configuration argument" not in str(e):
                    raise e
            mesh.unify_face_orientations()
            
            # Retry unwrapping on healed topology
            out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
                return_vmaps=True,
                verbose=verbose,
            )
            
        except BaseException as fallback_unwrap_err:
            print(f"[Fatal] Aggressive topological fallback failed to parameterize UVs: {fallback_unwrap_err}")
            local_cleanup()
            raise fallback_unwrap_err

    # Precompute vertex normal indices for correct rendering outputs
    try:
        out_vertices = out_vertices.cuda()
        out_faces = out_faces.cuda()
        out_uvs = out_uvs.cuda()
        out_vmaps = out_vmaps.cuda()
        
        mesh.compute_vertex_normals()
        out_normals = mesh.read_vertex_normals()[out_vmaps]
        
    except BaseException as normals_err:
        print(f"[Fatal] Normal computation or index mapping failed post-unwrapping: {normals_err}")
        local_cleanup()
        raise normals_err

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # ------------------------------------------------------------------------
    # Step 7: Differentiable Rasterization (nvdiffrast)
    # ------------------------------------------------------------------------
    # Projects the 2D UV layout into a 3D physical position coordinate space,
    # mapping every texture pixel (texel) to its exact 3D coordinate.
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)

    try:
        ctx = dr.RasterizeCudaContext()
        # Map normalized 0-1 UV coordinate channels to -1 to 1 clip coordinates
        uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
        
        # Rasterize in chunks to save memory
        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = dr.rasterize(
                ctx, uvs_rast, out_faces[i:i+100000],
                resolution=[texture_size, texture_size],
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i # Encode and store face index matching into alpha channels
            rast = torch.where(mask_chunk, rast_chunk, rast)
        
        mask = rast[0, ..., 3] > 0
        
        # Interpolate spatial coordinates within UV space
        pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
        valid_pos = pos[mask]
        
    except BaseException as rast_err:
        print(f"[Fatal] Differentiable rasterization (baking pass) failed: {rast_err}")
        local_cleanup()
        raise rast_err

    # ------------------------------------------------------------------------
    # Step 8: Spatial Projection & Attribute Queries (cuBVH Tree Lookup)
    # ------------------------------------------------------------------------
    # Map each simplified vertex back onto the continuous surface of the high-res 
    # original mesh via BVH. This removes color sliding artifacts.
    try:
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]] 
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        
    except BaseException as bvh_query_err:
        print(f"[Fatal] cuBVH spatial distance mapping projection query failed: {bvh_query_err}")
        local_cleanup()
        raise bvh_query_err

    # ------------------------------------------------------------------------
    # Step 9: Volumetric Trilinear Sampling (Grid sample)
    # ------------------------------------------------------------------------
    # Sample the RGB colors and material features (metallic, roughness, alpha)
    # directly from our 3D sparse latent volume at each projected coordinate.
    try:
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )
        
    except BaseException as sample_err:
        print(f"[Fatal] Volumetric trilinear attribute sampling failed: {sample_err}")
        local_cleanup()
        raise sample_err

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # ------------------------------------------------------------------------
    # Step 10: PBR Attribute Channels Extraction & Dilation Inpainting
    # ------------------------------------------------------------------------
    # Extract raw float attribute arrays, scale to 0-255 uint8 arrays, and 
    # inpaint (dilate) background edges to prevent black seam lines at UV margins.
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)

    try:
        mask = mask.cpu().numpy()
        
        # Extract channels based on configured layout parameters (RGB + Metallic + Roughness + Alpha)
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha_mode = 'OPAQUE'
        
        # Inpaint background margins using fast marching method to prevent black seam lines at UV margins
        mask_inv = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        
    except BaseException as img_proc_err:
        print(f"[Fatal] Image post-processing or inpainting failed: {img_proc_err}")
        local_cleanup()
        raise img_proc_err

    # ------------------------------------------------------------------------
    # Step 11: Material Creation & Trimesh Compilation
    # ------------------------------------------------------------------------
    # Pack channels into a PBRGLB-compatible material and build the Trimesh model.
    # We swap Y/Z axes and flip V coordinates here to match the standard GLB spec.
    try:
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode,
            doubleSided=True if not remesh else False,
        )
        
        # Move output attributes to numpy arrays
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
        
        textured_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        )
        
    except BaseException as trimesh_err:
        print(f"[Fatal] Trimesh creation or material mapping failed: {trimesh_err}")
        local_cleanup()
        raise trimesh_err

    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    # Flush caches
    local_cleanup()
    
    return textured_mesh

# Dynamically patch o_voxel.postprocess to use our robust method
o_voxel.postprocess.to_glb = robust_to_glb
