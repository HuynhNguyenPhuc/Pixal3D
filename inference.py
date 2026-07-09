import os
import sys
import math
import time
import warnings
import argparse
import shutil
import cv2
import torch
import numpy as np
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# System configuration environment variables
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "sdpa")

# Configure FlexGEMM Autotune cache to prevent permission issues across users
def configure_autotune_cache():
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    uid = os.getuid() if hasattr(os, 'getuid') else 0
    cache_name = f'autotune_cache_{uid}.json' if uid != 0 else 'autotune_cache.json'
    cache_path = os.path.join(cache_dir, cache_name)
    
    if uid != 0 and not os.path.exists(cache_path):
        base_cache = os.path.join(cache_dir, 'autotune_cache.json')
        if os.path.exists(base_cache):
            try:
                shutil.copy(base_cache, cache_path)
            except Exception:
                pass
                
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = cache_path
    os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '0'

configure_autotune_cache()

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

# ============================================================================
# Constants & Configurations
# ============================================================================
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"

# Dinosaur V3 Image Conditioning models configuration
IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Helpers
# ============================================================================
def build_image_cond_model(config: dict):
    """Builds and returns a Dinosaur V3 projection feature extractor."""
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device: str = "cuda", model_name: str = MOGE_MODEL_NAME):
    """Loads and prepares the MoGe depth/camera estimation model."""
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path: str = MODEL_PATH, device: str = "cuda", low_vram: bool = False):
    """Initializes and configures the generation pipeline."""
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    # Load NAF upsampler weights
    cond_attrs = [
        'image_cond_model_ss', 
        'image_cond_model_shape_512',
        'image_cond_model_shape_1024', 
        'image_cond_model_tex_1024'
    ]
    
    if low_vram:
        # Low-VRAM mode: Keep models on CPU, load them to GPU on-demand
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in cond_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: Move all components to GPU upfront
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in cond_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models loaded to GPU).")

    return pipeline

# ============================================================================
# Camera / Geometry Calibration
# ============================================================================
def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    """Computes the focal length in pixel units based on FOV and image resolution."""
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x: float, grid_point: torch.Tensor, target_point: torch.Tensor, mesh_scale: float, image_resolution: int) -> dict:
    """Calculates the camera distance to map grid coordinates to image coordinates."""
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    
    xw, yw, _ = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path: str, moge_model, device: str = "cuda", mesh_scale: float = 1.0, extend_pixel: int = 0, image_resolution: int = 512) -> dict:
    """Estimates camera intrinsics (FOV and Distance) using the MoGe model."""
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
        
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference Execution Pipeline
# ============================================================================
def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: bool = False,
    resolution: int = -1,
    no_webp: bool = True,
    decimation_target: int = 200000,
):
    # Dynamic VRAM Check & Mode Auto-switching
    if not low_vram and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            if free_mem / (1024**3) < 22.0:
                low_vram = True
                print(f"[Memory] Low free GPU memory detected ({free_mem / (1024**3):.2f} GB free). Enabling Low-VRAM mode.")
        except Exception as e:
            print(f"[Memory] Failed to verify GPU memory limit: {e}")

    # Load 3D Generation Pipeline
    pipeline = init_pipeline(model_path, low_vram=low_vram)

    # Preprocess Image
    print(f"[Inference] Preprocessing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)

    # Save a temporary preprocessed copy for the camera estimation model
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_preprocessed.save(tmp_path)

    # Camera Calibration
    if manual_fov > 0:
        camera_angle_x = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
            mesh_scale, image_resolution
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}
        print(f"[Inference] Using manual FOV: {math.degrees(manual_fov):.2f}° ({manual_fov:.4f} rad), distance={distance:.4f}")
    else:
        print("[MoGe-2] Initializing model for camera prediction...")
        moge_model = load_moge_model(device="cuda")
        print("[Inference] Estimating camera parameters via MoGe-2...")
        camera_params = get_camera_params_wild_moge(
            tmp_path, moge_model, device="cuda",
            mesh_scale=mesh_scale, extend_pixel=extend_pixel,
            image_resolution=image_resolution,
        )
        print(f"[Inference] Calibration: camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")
        
        # Offload MoGe-2 from GPU to free memory for cascade stages
        moge_model.cpu()
        del moge_model
        torch.cuda.empty_cache()
        
    os.remove(tmp_path)

    # Synthesize Mesh Latents
    print("[Inference] Running cascade flow matching pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if low_vram else 1536)}_cascade"
    print(f"[Inference] Stage execution target: {pipeline_type}")
    
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )
    mesh = mesh_list[0]

    # Free heavy sparse latents that are no longer needed for triangulation to save massive GPU VRAM.
    del shape_slat, tex_slat
    from utilities.gpu import aggressive_gpu_cleanup, simplify_mesh, clean_mesh
    aggressive_gpu_cleanup()

    try:
        mesh_vertices = mesh.vertices
        mesh_faces = mesh.faces

        # Always run GPU-based clean to prevent CuMesh illegal memory access on degenerate meshes
        mesh_vertices, mesh_faces = clean_mesh(mesh_vertices, mesh_faces)

        import config
        if simplify_mesh is not None and mesh_faces.shape[0] >= config.SIMPLIFICATION_THRESHOLD_FACES:
            print(f"[Inference] Proactive Safeguard: Mesh has >= {config.SIMPLIFICATION_THRESHOLD_FACES:,} faces ({mesh_faces.shape[0]:,} faces). Simplifying to {config.SIMPLIFICATION_TARGET_FACES:,} faces on GPU to prevent OOM...")
            try:
                mesh_vertices, mesh_faces = simplify_mesh(mesh_vertices, mesh_faces, config.SIMPLIFICATION_TARGET_FACES)
                print(f"[Inference] Proactive GPU Simplification complete. New face count: {mesh_faces.shape[0]:,}")
            except BaseException as e:
                print(f"[Inference] Proactive GPU Simplification failed: {type(e).__name__} - {e}. Proceeding with original density.")

        # Extract & Triangulate GLB
        print(f"[Inference] Extracting GLB mesh (Grid resolution: {res})...")
        disable_tqdm = os.environ.get("DISABLE_TQDM", "0") == "1"
        
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh_vertices, faces=mesh_faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target, texture_size=4096,
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=not disable_tqdm,
        )

        # Apply 180 degrees frontal rotation around Y-axis
        rot = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1],
        ], dtype=np.float64)
        glb.apply_transform(rot)

        # Save to final output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        glb.export(output_path, extension_webp=not no_webp)
        print(f"[Done] Export complete! GLB saved to: {output_path}")

    finally:
        aggressive_gpu_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage. "
                             "Reduces peak VRAM from ~18GB to ~10-12GB at the cost of slower inference.")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram, else 1536.")
    parser.add_argument("--no_webp", action="store_true",
                        help="Disable WebP texture compression in exported GLB (uses standard PNG instead).")
    parser.add_argument("--decimation_target", type=int, default=200000,
                        help="Target number of faces for mesh decimation. Default is 200,000.")

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=args.resolution,
        no_webp=args.no_webp,
        decimation_target=args.decimation_target,
    )
