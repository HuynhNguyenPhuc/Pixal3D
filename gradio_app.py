import os
import sys
import gc
import math
import time
import json
import shutil
import random
import asyncio
import mimetypes
import warnings
import argparse
import threading
import trimesh
import subprocess as _sp
from typing import Dict, List, Tuple, Any, Optional

import cv2
import torch
import numpy as np
from PIL import Image

import config

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Global System Settings
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

# Gradio & FastAPI Web Stack Imports
import gradio as gr
from gradio import Server
from gradio.data_classes import FileData
from fastapi import Response, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Lock for model initialization safety
init_lock = threading.Lock()

# Custom static files loader with caching control headers
class CacheStaticFiles(StaticFiles):
    def __init__(self, *args, cache_control: str = None, **kwargs):
        self.cache_control = cache_control
        super().__init__(*args, **kwargs)

    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        if self.cache_control:
            response.headers["Cache-Control"] = self.cache_control
        return response

from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.renderers import EnvMap
from pixal3d.utils import render_utils
import o_voxel

# ============================================================================
# Constants & Configurations
# ============================================================================
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# Preview lighting/render modes configured on the UI
MODES = [
    {"name": "Normal", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "Clay render", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "Base color", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI forest", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8

# Cascade parameters
CASCADE_LR_RESOLUTION = 512
CASCADE_MAX_NUM_TOKENS = 49152

# MoGe parameters
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
WILD_MESH_SCALE = 1.0
WILD_EXTEND_PIXEL = 0
WILD_IMAGE_RESOLUTION = 512

# Dinosaur V3 Feature Extractor models
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
# Lazy-Loaded Model Globals & Setup
# ============================================================================
pipeline = None
moge_model = None
envmap = None
LOW_VRAM = os.environ.get("LOW_VRAM", "0") == "1"

# Automatic low-vram detector for systems with tight hardware resources
if not LOW_VRAM and torch.cuda.is_available():
    try:
        free_mem, _ = torch.cuda.mem_get_info()
        # If less than 22 GB free, auto-enable low_vram mode to prevent OOMs during cascade
        if free_mem / (1024**3) < 22.0:
            LOW_VRAM = True
            print(f"[Memory] Low free GPU memory detected ({free_mem / (1024**3):.2f} GB free). Auto-enabling Low-VRAM mode.")
    except Exception as e:
        print(f"[Memory] Failed to verify system GPU memory: {e}")


def build_image_cond_model(config: dict):
    """Builds and returns a Dinosaur V3 projection feature extractor."""
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device: str = "cuda", model_name: str = MOGE_MODEL_NAME):
    """Loads and returns the MoGe model."""
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model


def build_thumbnails():
    """Generates fast, lightweight preview thumbnails for the sample gallery on startup."""
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'images')
    thumb_dir = os.path.join(img_dir, 'thumbnails')
    os.makedirs(thumb_dir, exist_ok=True)
    
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not os.path.isdir(os.path.join(img_dir, f)):
            src_path = os.path.join(img_dir, f)
            name_without_ext = os.path.splitext(f)[0]
            dest_path = os.path.join(thumb_dir, f"{name_without_ext}.webp")
            
            if os.path.exists(dest_path) and os.path.getmtime(dest_path) > os.path.getmtime(src_path):
                continue
                
            try:
                img = Image.open(src_path)
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                img.save(dest_path, 'WEBP', quality=85)
                print(f"[Thumbnails] Generated thumbnail: {f} -> webp")
            except Exception as e:
                print(f"[Thumbnails] Failed to generate thumbnail for {f}: {e}")


def init_models():
    """Thread-safe lazy initialization of generation pipelines and models."""
    global pipeline, moge_model, envmap
    with init_lock:
        if pipeline is not None:
            return

        # Print GPU environment diagnostics
        print("=" * 60)
        print("[Diagnostics] PyTorch version:", torch.__version__)
        print("[Diagnostics] CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("[Diagnostics] CUDA version:", torch.version.cuda)
            print("[Diagnostics] cuDNN version:", torch.backends.cudnn.version())
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"[Diagnostics] GPU {i}: {name}, sm_{cap[0]}{cap[1]}, {mem:.1f} GB")
        try:
            res = _sp.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            print("[Diagnostics] nvidia-smi:", res.stdout.strip())
        except Exception as e:
            print(f"[Diagnostics] nvidia-smi failed: {e}")
        print("=" * 60)

        model_path = "TencentARC/Pixal3D"
        print(f"[Pipeline] Loading from {model_path}...")
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
        
        print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
        pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
        pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
        pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
        pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])
        
        cond_attrs = [
            'image_cond_model_ss', 
            'image_cond_model_shape_512',
            'image_cond_model_shape_1024', 
            'image_cond_model_tex_1024'
        ]

        if LOW_VRAM:
            # Low-VRAM mode: keep conditioning extractors on CPU, move to GPU as needed
            print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
            for attr in cond_attrs:
                m = getattr(pipeline, attr, None)
                if m is not None and getattr(m, 'use_naf_upsample', False):
                    m._load_naf()
            pipeline._device = torch.device("cuda")
            pipeline.low_vram = True
            print("[Pipeline] Low-VRAM mode enabled.")
        else:
            # Move everything directly onto the GPU
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
                
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cpu")
        print("[MoGe-2] Loaded on CPU (VRAM reserved exclusively for generation).")
        
        print("[EnvMap] Loading high dynamic range environment maps...")
        _base = os.path.dirname(os.path.abspath(__file__))
        _envmap_device = 'cpu' if LOW_VRAM else 'cuda'
        envmap = {
            'forest': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
            'sunset': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/sunset.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
            'courtyard': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/courtyard.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
        }

# ============================================================================
# Camera & Geometry Calibration
# ============================================================================
def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    """Computes camera focal length in pixel units."""
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x: float, grid_point: torch.Tensor, target_point: torch.Tensor, mesh_scale: float, image_resolution: int) -> dict:
    """Calculates ideal projection camera distance to map grid to pixel coordinates."""
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    
    xw, yw, _ = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path: str, device: str = "cuda", mesh_scale: float = 1.0, extend_pixel: int = 0, image_resolution: int = 512) -> dict:
    """Estimates optimal camera properties on an uncalibrated input image using MoGe."""
    global moge_model
    if moge_model is None:
        print("[MoGe-2] Lazy-loading MoGe-2 model...")
        moge_model = load_moge_model(device="cpu")
        
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    
    print("[MoGe-2] Offloading camera estimation to GPU...")
    moge_model.to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
        
    print("[MoGe-2] Moving model back to CPU to free VRAM...")
    moge_model.cpu()
    torch.cuda.empty_cache()
    
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
# Serialization & Latent State I/O
# ============================================================================
def resolve_state_path(state_path: str) -> str:
    """Resolves relative or web-accessible paths to local folder storage paths."""
    if "/tmp/" in state_path or "\\tmp\\" in state_path:
        parts = state_path.replace("\\", "/").split("/tmp/")
        return os.path.join(TMP_DIR, parts[-1])
    elif "model_" in state_path and not state_path.startswith(TMP_DIR):
        parts = state_path.split("model_")
        return os.path.join(TMP_DIR, "model_" + parts[-1])
    return os.path.abspath(state_path)


def pack_state(shape_slat, tex_slat, res: int, model_id: str = None) -> str:
    """Compresses generated latents and coordinate mappings into a single NPZ archive."""
    state_data = {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    if model_id is None:
        model_id = f"state_{int(time.time()*1000)}_{random.randint(0,9999):04d}"
    job_dir = os.path.join(TMP_DIR, model_id)
    os.makedirs(job_dir, exist_ok=True)
    state_path = os.path.join(job_dir, "state.npz")
    np.savez_compressed(state_path, **state_data)
    return state_path


def unpack_state(state_path: str) -> Tuple[SparseTensor, SparseTensor, int]:
    """Loads and reconstructs SparseTensor structures from compressed state files."""
    state_path = resolve_state_path(state_path)
    data = np.load(state_path)
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(data['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, int(data['res'])

# ============================================================================
# History & Directory Lifecycle Manager
# ============================================================================
class HistoryManager:
    def __init__(self, limit: int = 5):
        self.limit = limit
        self.history_file = os.path.join(TMP_DIR, "history.json")
        self.lock = threading.Lock()
        self.models = self._load()
        self._sync_with_disk()

    def _load(self) -> List[dict]:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _sync_with_disk(self):
        """Deduplicates and cleans metadata records to sync with disk directories."""
        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR, exist_ok=True)
            return

        with self.lock:
            valid_models = []
            changed = False
            for model in self.models:
                model_id = model.get("id")
                if model_id:
                    job_dir = os.path.join(TMP_DIR, model_id)
                    if os.path.exists(job_dir):
                        valid_models.append(model)
                    else:
                        print(f"[HistoryManager] Evicting missing directory: {model_id} from metadata.")
                        changed = True
                else:
                    changed = True

            self.models = valid_models
            if changed:
                self._save()

            # Safely clear folders that are not registered inside metadata
            try:
                allowed_ids = {m["id"] for m in self.models if "id" in m}
                for item in os.listdir(TMP_DIR):
                    item_path = os.path.join(TMP_DIR, item)
                    if os.path.isdir(item_path) and item.startswith("model_"):
                        if item not in allowed_ids:
                            print(f"[HistoryManager] Pruning unregistered directory: {item}")
                            shutil.rmtree(item_path)
            except Exception as e:
                print(f"[HistoryManager] Error purging unregistered directories: {e}")

    def _save(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            print(f"[HistoryManager] Failed to write history file: {e}")

    def add_generation(self, model_id: str, image_path: str, state_path: str, renders_dict: dict):
        with self.lock:
            existing = next((m for m in self.models if m['id'] == model_id), None)
            
            # Select appropriate thumbnails
            thumbnail_url = ""
            if renders_dict and 'shaded_forest' in renders_dict:
                first_render = renders_dict['shaded_forest'][0]
                if isinstance(first_render, dict) and 'path' in first_render:
                    filename = os.path.basename(first_render['path'])
                    thumbnail_url = f"/tmp/{model_id}/{filename}"
                elif hasattr(first_render, 'path'):
                    thumbnail_url = f"/tmp/{model_id}/{os.path.basename(first_render.path)}"
                elif isinstance(first_render, str):
                    thumbnail_url = f"/tmp/{model_id}/{os.path.basename(first_render)}"

            if not thumbnail_url and renders_dict:
                # Fallback to any computed render
                for mode, frames in renders_dict.items():
                    if frames:
                        f = frames[0]
                        if isinstance(f, dict) and 'path' in f:
                            filename = os.path.basename(f['path'])
                        elif hasattr(f, 'path'):
                            filename = os.path.basename(f.path)
                        else:
                            filename = os.path.basename(str(f))
                        thumbnail_url = f"/tmp/{model_id}/{filename}"
                        break

            image_url = f"/tmp/{model_id}/image.png"

            entry = {
                "id": model_id,
                "timestamp": int(time.time() * 1000),
                "image_url": image_url,
                "state_path": os.path.abspath(state_path),
                "thumbnail_url": thumbnail_url,
                "glb_url": existing.get("glb_url", "") if existing else "",
                "status": "generated"
            }

            if existing:
                self.models.remove(existing)
            self.models.insert(0, entry)
            self._save()

    def add_preprocess(self, model_id: str, image_path: str):
        with self.lock:
            existing = next((m for m in self.models if m['id'] == model_id), None)
            if existing and existing.get("status") in ("generated", "completed"):
                return
            
            entry = {
                "id": model_id,
                "timestamp": int(time.time() * 1000),
                "image_url": f"/tmp/{model_id}/image.png",
                "state_path": "",
                "thumbnail_url": "",
                "glb_url": "",
                "status": "preprocessed"
            }
            if existing:
                self.models.remove(existing)
            self.models.insert(0, entry)
            self._save()

    def add_glb(self, model_id: str, glb_path: str):
        with self.lock:
            existing = next((m for m in self.models if m['id'] == model_id), None)
            if not existing:
                existing = {
                    "id": model_id,
                    "timestamp": int(time.time() * 1000),
                    "image_url": "",
                    "state_path": "",
                    "thumbnail_url": "",
                    "glb_url": ""
                }
                self.models.insert(0, existing)

            existing["glb_url"] = f"/tmp/{model_id}/result.glb"
            existing["status"] = "completed"
            
            # LRU Metadata & File Eviction
            while len(self.models) > self.limit:
                oldest = self.models.pop()
                self._delete_model_files(oldest)

            self._save()

    def _delete_model_files(self, model_entry: dict):
        model_id = model_entry['id']
        print(f"[HistoryManager] Reached LRU limit. Evicting assets for: {model_id}...")
        job_dir = os.path.join(TMP_DIR, model_id)
        if os.path.exists(job_dir):
            try:
                shutil.rmtree(job_dir)
                print(f"[HistoryManager] Cleaned job directory: {job_dir}")
            except Exception as e:
                print(f"[HistoryManager] Error cleaning job directory {job_dir}: {e}")
                
        # Auto clean orphan or loose files in TMP_DIR older than 1 hour
        try:
            now = time.time()
            for f in os.listdir(TMP_DIR):
                f_path = os.path.join(TMP_DIR, f)
                if os.path.isfile(f_path) and (f_path.endswith('.png') or f_path.endswith('.json')) and (now - os.path.getmtime(f_path) > 3600):
                    os.remove(f_path)
        except Exception as e:
            print(f"[HistoryManager] Background temp disk cleanup error: {e}")

    def get_all(self) -> List[dict]:
        with self.lock:
            return [m for m in self.models]


history_manager = HistoryManager(limit=50)

# ============================================================================
# Progress Interception & Monkeypatching
# ============================================================================
_thread_local = threading.local()


def _progress_file(model_id: str) -> str:
    """Gets the relative path of progress logging configuration inside Job directory."""
    job_dir = os.path.join(TMP_DIR, model_id)
    os.makedirs(job_dir, exist_ok=True)
    return os.path.join(job_dir, "progress.json")


def _reset_progress(model_id: str):
    _thread_local.active_model_id = model_id
    _write_progress_file(model_id, {"stage": "Initializing...", "step": 0, "total": 0, "done": False})


def _update_progress(stage: str, step: int, total: int):
    model_id = getattr(_thread_local, 'active_model_id', '')
    if model_id:
        _write_progress_file(model_id, {"stage": stage, "step": step, "total": total, "done": False})


def _finish_progress():
    model_id = getattr(_thread_local, 'active_model_id', '')
    if model_id:
        _write_progress_file(model_id, {"done": True})


def _write_progress_file(model_id: str, data: dict):
    """Write updates atomically to progress configurations file."""
    path = _progress_file(model_id)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception:
        pass


import tqdm as _tqdm_module
_original_tqdm = _tqdm_module.tqdm

class _TqdmProgressInterceptor(_original_tqdm):
    """Intercepts and translates tqdm iterations directly to Gradio Polling updates."""
    def __init__(self, *args, **kwargs):
        self._stage_desc = kwargs.get('desc', 'Processing')
        super().__init__(*args, **kwargs)
    
    def set_description(self, desc=None, refresh=True):
        self._stage_desc = desc or 'Processing'
        super().set_description(desc, refresh)
    
    def update(self, n=1):
        super().update(n)
        _update_progress(self._stage_desc, self.n, self.total or 0)


# Globally wrap tqdm with progress tracker
_tqdm_module.tqdm = _TqdmProgressInterceptor
import pixal3d.pipelines.samplers.flow_euler as _fe_module
_fe_module.tqdm = _TqdmProgressInterceptor
import pixal3d.utils.render_utils as _ru_module
_ru_module.tqdm = _TqdmProgressInterceptor
import o_voxel.postprocess as _ovp_module
_ovp_module.tqdm = _TqdmProgressInterceptor

# ============================================================================
# Server-side API & Web Router Implementation
# ============================================================================
app = Server()


@app.get("/")
async def homepage():
    """Serves the central user dashboard index file."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(
            content=f.read(),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )


@app.get("/sw.js")
async def service_worker():
    """Provides self-uninstalling service worker registration to clear old caching problems."""
    sw_code = """
self.addEventListener('install', (event) => {
    self.skipWaiting();
});
self.addEventListener('activate', (event) => {
    self.registration.unregister().then(() => self.clients.claim());
});
    """
    return Response(content=sw_code, media_type="application/javascript")


@app.get("/app_config")
async def get_config():
    """Returns application initialization metadata to UI clients."""
    return JSONResponse({"low_vram": LOW_VRAM})


@app.get("/progress")
async def progress_poll(request: Request):
    """Polling target providing progress indicators to front-end components."""
    model_id = request.query_params.get("model_id", "")
    path = _progress_file(model_id)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return JSONResponse(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"stage": "Waiting...", "step": 0, "total": 0, "done": False})


@app.get("/history")
async def get_history():
    """Fetches generation catalogs for the side panel dashboard viewer."""
    return JSONResponse(history_manager.get_all())


# ============================================================================
# Gradio Pipeline Operations Endpoint Handlers
# ============================================================================
@app.api()
def preprocess(image: FileData, model_id: str) -> FileData:
    """Preprocesses input assets, removes the background, and isolates geometries."""
    init_models()
    try:
        img = Image.open(image["path"])
        processed = pipeline.preprocess_image(img)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
            print("[OOM] CUDA Out of memory during preprocessing. Falling back to CPU...")
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            try:
                pipeline.rembg_model.cpu()
                img = Image.open(image["path"])
                processed = pipeline.preprocess_image(img)
                if torch.cuda.is_available():
                    pipeline.rembg_model.cuda()
            except Exception as cpu_err:
                print(f"[OOM] CPU Fallback failed: {cpu_err}")
                raise gr.Error("CUDA Out of Memory during preprocessing. Please downscale the input image.")
        else:
            raise e
    except Exception as e:
        print(f"[Preprocess Error] {e}")
        raise gr.Error(f"Preprocessing pipeline failed: {str(e)}")
    
    job_dir = os.path.join(TMP_DIR, model_id)
    os.makedirs(job_dir, exist_ok=True)
    out_path = os.path.join(job_dir, "image.png")
    processed.save(out_path)
    
    history_manager.add_preprocess(model_id, out_path)
    return FileData(path=out_path)


@app.api()
def generate_3d(
    seed: int, 
    resolution: int,
    model_id: str,
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
    manual_fov: float = -1.0,
    fov_unit: str = "deg",
) -> Dict:
    """Executes the cascade generation pipeline for structural latents."""
    init_models()
    _reset_progress(model_id)
    _update_progress("Preprocessing & Camera Estimation", 0, 1)
    
    torch.manual_seed(seed)
    hr_resolution = int(resolution)
    
    job_dir = os.path.join(TMP_DIR, model_id)
    os.makedirs(job_dir, exist_ok=True)
    
    image_copy_path = os.path.join(job_dir, "image.png")
    image_preprocessed = Image.open(image_copy_path)
    
    temp_processed_path = os.path.join(job_dir, "temp_proc.png")
    image_preprocessed.save(temp_processed_path)
    
    # Process camera properties
    if manual_fov > 0:
        if fov_unit == "rad":
            camera_angle_x = float(manual_fov)
            fov_deg = math.degrees(manual_fov)
        else:
            camera_angle_x = math.radians(manual_fov)
            fov_deg = float(manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = distance_from_fov(
            camera_angle_x, grid_point,
            torch.tensor([0 - WILD_EXTEND_PIXEL, WILD_IMAGE_RESOLUTION - 1 + WILD_EXTEND_PIXEL]),
            WILD_MESH_SCALE, WILD_IMAGE_RESOLUTION
        )["distance_from_x"]
        camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': WILD_MESH_SCALE}
        print(f"[Camera] Using manual FOV: {fov_deg:.2f}° ({camera_angle_x:.4f} rad), distance: {distance:.4f}")
    else:
        camera_params = get_camera_params_wild_moge(
            temp_processed_path, device="cuda",
            mesh_scale=WILD_MESH_SCALE, extend_pixel=WILD_EXTEND_PIXEL,
            image_resolution=WILD_IMAGE_RESOLUTION,
        )
    _update_progress("Preprocessing & Camera Estimation", 1, 1)
    
    ss_sampler_override = {"steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
                           "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t}
    shape_sampler_override = {"steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
                              "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t}
    tex_sampler_override = {"steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
                            "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t}

    pipeline_type = f"{hr_resolution}_cascade"
    try:
        # Run generation cascade
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
            max_num_tokens=CASCADE_MAX_NUM_TOKENS,
        )
        
        mesh = mesh_list[0]
        state_path = pack_state(shape_slat, tex_slat, res, model_id=model_id)
        
        # Free heavy sparse latents and other intermediate structures to maximize VRAM for rendering/export
        del shape_slat, tex_slat, mesh_list
        from utilities.gpu import aggressive_gpu_cleanup
        aggressive_gpu_cleanup()
        
        # Render previews of intermediate outputs
        _update_progress("Rendering views", 0, 1)
        mesh.simplify(16777216)
        cam_dist = camera_params['distance']
        near = max(0.01, cam_dist - 2.0)
        far = cam_dist + 10.0
        
        if LOW_VRAM:
            for v in envmap.values():
                v.image = v.image.cuda()
                if hasattr(v, '_nvdiffrec_envlight'):
                    del v._nvdiffrec_envlight
                    
        renders = render_utils.render_proj_aligned_video(
            mesh, camera_angle_x=camera_params['camera_angle_x'],
            distance=cam_dist, resolution=1024,
            num_frames=STEPS, envmap=envmap,
            near=near, far=far,
        )
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
            print("[OOM] CUDA Out of Memory during 3D generation. Purging cache...")
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            raise gr.Error("CUDA Out of Memory: Cascade ran out of VRAM. Try choosing a lower target resolution.")
        else:
            raise e
    except Exception as e:
        print(f"[Generation Error] {e}")
        raise gr.Error(f"3D Generation failed: {str(e)}")
        
    if LOW_VRAM:
        for v in envmap.values():
            if hasattr(v, '_nvdiffrec_envlight'):
                del v._nvdiffrec_envlight
            v.image = v.image.cpu()
        torch.cuda.empty_cache()
        
    _update_progress("Rendering views", 1, 1)
    
    # Save preview image files
    render_files = {}
    for mode_key, frames in renders.items():
        mode_files = []
        for i, frame in enumerate(frames):
            p = os.path.abspath(os.path.join(job_dir, f"render_{mode_key}_{i}.jpg"))
            Image.fromarray(frame).save(p, quality=85)
            mode_files.append(FileData(path=p))
        render_files[mode_key] = mode_files

    _finish_progress()
    
    res_dict = {
        "render_paths": render_files,
        "state_path": os.path.abspath(state_path),
        "camera_angle_x": camera_params['camera_angle_x'],
        "distance": camera_params['distance'],
    }
    
    history_manager.add_generation(model_id, image_copy_path, state_path, render_files)
    return res_dict


@app.api()
def extract_glb_api(state_path: str, decimation_target: int, texture_size: int, no_webp: bool = False, session_id: str = "") -> FileData:
    """Decodes latent codes and extracts a high-fidelity triangulated 3D mesh (GLB)."""
    init_models()
    _reset_progress(session_id)
    _update_progress("Decoding latent", 0, 1)
    
    try:
        shape_slat, tex_slat, res = unpack_state(state_path)
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        _update_progress("Decoding latent", 1, 1)
        
        # Free heavy sparse latents that are no longer needed for triangulation to save massive GPU VRAM.
        del shape_slat, tex_slat
        from utilities.gpu import aggressive_gpu_cleanup, simplify_mesh, clean_mesh
        aggressive_gpu_cleanup()
        
        mesh_vertices = mesh.vertices
        mesh_faces = mesh.faces

        # Always run GPU-based clean to prevent CuMesh illegal memory access on degenerate meshes
        mesh_vertices, mesh_faces = clean_mesh(mesh_vertices, mesh_faces)

        if mesh_faces.shape[0] >= config.SIMPLIFICATION_THRESHOLD_FACES:
            print(f"Proactive Safeguard: Mesh has >= {config.SIMPLIFICATION_THRESHOLD_FACES:,} faces ({mesh_faces.shape[0]:,} faces). Simplifying to {config.SIMPLIFICATION_TARGET_FACES:,} faces on GPU to prevent OOM...")
            try:
                mesh_vertices, mesh_faces = simplify_mesh(mesh_vertices, mesh_faces, config.SIMPLIFICATION_TARGET_FACES)
                print(f"Proactive GPU Simplification complete. New face count: {mesh_faces.shape[0]:,}")
            except BaseException as e:
                print(f"Proactive GPU Simplification failed: {type(e).__name__} - {e}. Proceeding with original density.")

        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_vertices, faces=mesh_faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
                grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target, texture_size=texture_size,
                remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            is_oom = "out of memory" in exc_str or "cuda error" in exc_str or isinstance(exc, torch.OutOfMemoryError)
            
            if is_oom:
                print("[OOM] OOM during to_glb remesh. Retrying without remesh...")
                aggressive_gpu_cleanup()
                
                try:
                    glb = o_voxel.postprocess.to_glb(
                        vertices=mesh_vertices, faces=mesh_faces, attr_volume=mesh.attrs,
                        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
                        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=decimation_target, texture_size=texture_size,
                        remesh=False, remesh_band=1, remesh_project=0, use_tqdm=True,
                    )
                except Exception as exc_fallback:
                    exc_fallback_str = str(exc_fallback).lower()
                    is_fallback_oom = "out of memory" in exc_fallback_str or "cuda error" in exc_fallback_str or isinstance(exc_fallback, torch.OutOfMemoryError)
                    
                    if is_fallback_oom:
                        print("[OOM] OOM during to_glb fallback. Attempting GPU-decimation fallback and lowering texture size...")
                        aggressive_gpu_cleanup()
                        
                        gpu_target = max(decimation_target, 200000)
                        
                        try:
                            print(f"Simplifying mesh on GPU to {gpu_target:,} faces...")
                            mesh_vertices_fallback, mesh_faces_fallback = simplify_mesh(mesh_vertices, mesh_faces, gpu_target)
                        except BaseException as e_gpu:
                            print(f"Fallback GPU Simplification failed: {type(e_gpu).__name__} - {e_gpu}. Proceeding with original density.")
                            mesh_vertices_fallback, mesh_faces_fallback = mesh_vertices, mesh_faces
                        
                        print("Retrying to_glb on simplified mesh (no remesh, 1024 texture)...")
                        glb = o_voxel.postprocess.to_glb(
                            vertices=mesh_vertices_fallback, faces=mesh_faces_fallback, attr_volume=mesh.attrs,
                            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
                            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=decimation_target, texture_size=min(texture_size, 1024),
                            remesh=False, remesh_band=1, remesh_project=0, use_tqdm=True,
                        )
                    else:
                        raise exc_fallback
            else:
                raise exc
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
            print("[OOM] CUDA Out of Memory during GLB extraction. Purging cache...")
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            raise gr.Error("CUDA Out of Memory: Mesh triangulation ran out of memory. Try lowering the texture size limit.")
        else:
            raise e
    except Exception as e:
        print(f"[GLB Extraction Error] {e}")
        raise gr.Error(f"GLB Extraction failed: {str(e)}")
        
    # Rotate mesh by 180 degrees frontal orientation around Y-axis
    rot = np.array([
        [ 1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    
    resolved_state_path = resolve_state_path(state_path)
    job_dir = os.path.dirname(resolved_state_path)
    model_id = os.path.basename(job_dir)
        
    out_glb = os.path.join(job_dir, "result.glb")
    glb.export(out_glb, extension_webp=not no_webp)
    
    history_manager.add_glb(model_id, out_glb)
    _finish_progress()
    return FileData(path=out_glb)


# ============================================================================
# Static Files mounting & execution startup wrapper
# ============================================================================
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", CacheStaticFiles(directory="assets", cache_control="public, max-age=31536000, immutable"), name="assets")
app.mount("/tmp", CacheStaticFiles(directory=TMP_DIR, cache_control="public, max-age=86400"), name="tmp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Demo Server")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models lazy-load to GPU per stage.")
    args, remaining = parser.parse_known_args()
    if args.low_vram:
        LOW_VRAM = True

    # Pre-generate lightweight fast gallery thumbnails
    build_thumbnails()
    
    # Initialize networks
    init_models()
    
    app.launch(show_error=True, share=True)
