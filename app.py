import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import subprocess
import argparse
import math
import time
import shutil
import cv2
import torch
import numpy as np
import base64
import io
import json
from datetime import datetime
from typing import *
from PIL import Image

import threading
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Lock for model initialization
init_lock = threading.Lock()

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn_3")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '0'

import gradio as gr
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Response

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
# Constants & Defaults
# ============================================================================

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

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

# MoGe defaults
MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
WILD_MESH_SCALE = 1.0
WILD_EXTEND_PIXEL = 0
WILD_IMAGE_RESOLUTION = 512

# Image Cond Model configs
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
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model

def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model

def build_thumbnails():
    """Build fast, lightweight webp thumbnails on startup for the sample gallery."""
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'images')
    thumb_dir = os.path.join(img_dir, 'thumbnails')
    os.makedirs(thumb_dir, exist_ok=True)
    
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not os.path.isdir(os.path.join(img_dir, f)):
            src_path = os.path.join(img_dir, f)
            name_without_ext = os.path.splitext(f)[0]
            dest_path = os.path.join(thumb_dir, f"{name_without_ext}.webp")
            
            # If thumbnail already exists and is newer than source, skip
            if os.path.exists(dest_path) and os.path.getmtime(dest_path) > os.path.getmtime(src_path):
                continue
                
            try:
                img = Image.open(src_path)
                # Keep aspect ratio but scale down so max dimension is 256
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                # Save as high-quality WebP
                img.save(dest_path, 'WEBP', quality=85)
                print(f"[Thumbnails] Generated optimized thumbnail: {f} -> webp")
            except Exception as e:
                print(f"[Thumbnails] Failed to generate thumbnail for {f}: {e}")

# Global instances (lazy loaded or loaded at start)
pipeline = None
moge_model = None
envmap = None
LOW_VRAM = os.environ.get("LOW_VRAM", "0") == "1"

if not LOW_VRAM and torch.cuda.is_available():
    try:
        free_mem, total_mem = torch.cuda.mem_get_info()
        # If less than 22 GB free, auto-enable low_vram mode to prevent OOMs
        if free_mem / (1024**3) < 22.0:
            LOW_VRAM = True
            print(f"[Memory] Low free GPU memory detected ({free_mem / (1024**3):.2f} GB free). Automatically enabling Low-VRAM mode.")
    except Exception as e:
        print(f"[Memory] Failed to check GPU memory: {e}")

def init_models():
    global pipeline, moge_model, envmap
    with init_lock:
        if pipeline is not None:
            return

        # GPU / CUDA Diagnostics (runs when GPU is allocated)
        import subprocess as _sp
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
        
        if LOW_VRAM:
            # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
            print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
            for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                         'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
                m = getattr(pipeline, attr, None)
                if m is not None and getattr(m, 'use_naf_upsample', False):
                    m._load_naf()
            pipeline._device = torch.device("cuda")
            pipeline.low_vram = True
            print("[Pipeline] Low-VRAM mode enabled.")
        else:
            # Standard mode: all models loaded to GPU at once.
            pipeline.low_vram = False
            pipeline.cuda()
            pipeline.image_cond_model_ss.cuda()
            pipeline.image_cond_model_shape_512.cuda()
            pipeline.image_cond_model_shape_1024.cuda()
            pipeline.image_cond_model_tex_1024.cuda()
            print("[NAF] Pre-loading NAF upsampler model...")
            for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                         'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
                m = getattr(pipeline, attr, None)
                if m is not None and getattr(m, 'use_naf_upsample', False):
                    m._load_naf()
                
        print("[MoGe-2] Loading model for camera estimation...")
        moge_model = load_moge_model(device="cpu")
        print("[MoGe-2] Loaded on CPU (VRAM kept free for pipeline).")
        
        print("[EnvMap] Loading environment maps...")
        _base = os.path.dirname(os.path.abspath(__file__))
        _envmap_device = 'cpu' if LOW_VRAM else 'cuda'
        envmap = {
            'forest': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
            'sunset': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/sunset.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
            'courtyard': EnvMap(torch.tensor(cv2.cvtColor(cv2.imread(os.path.join(_base, 'assets/hdri/courtyard.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), dtype=torch.float32, device=_envmap_device)),
        }

# ============================================================================
# Utilities
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())

def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}

def get_camera_params_wild_moge(image_path, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    global moge_model
    if moge_model is None:
        print("[MoGe-2] Lazy-loading MoGe-2 model...")
        moge_model = load_moge_model(device="cpu")
        
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    
    print("[MoGe-2] Moving to GPU for estimation...")
    moge_model.to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
        
    print("[MoGe-2] Offloading from GPU to CPU to keep VRAM 100% free for cascade...")
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

def pack_state(shape_slat, tex_slat, res, model_id=None):
    state_data = {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    if model_id is None:
        import random
        model_id = f"state_{int(time.time()*1000)}_{random.randint(0,9999):04d}"
    job_dir = os.path.join(TMP_DIR, model_id)
    os.makedirs(job_dir, exist_ok=True)
    state_path = os.path.join(job_dir, "state.npz")
    np.savez_compressed(state_path, **state_data)
    return state_path

class HistoryManager:
    def __init__(self, limit=5):
        self.limit = limit
        self.history_file = os.path.join(TMP_DIR, "history.json")
        self.lock = threading.Lock()
        self.models = self._load()
        self._sync_with_disk()

    def _load(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _sync_with_disk(self):
        """Synchronize history with actual folders in TMP_DIR.
        Remove missing models from history, and delete orphaned directories in TMP_DIR.
        """
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
                        print(f"[HistoryManager] Directory for {model_id} does not exist. Removing from history.")
                        changed = True
                else:
                    changed = True

            self.models = valid_models
            if changed:
                self._save()

            try:
                allowed_ids = {m["id"] for m in self.models if "id" in m}
                for item in os.listdir(TMP_DIR):
                    item_path = os.path.join(TMP_DIR, item)
                    if os.path.isdir(item_path) and item.startswith("model_"):
                        if item not in allowed_ids:
                            print(f"[HistoryManager] Found orphaned directory {item}. Deleting...")
                            import shutil
                            shutil.rmtree(item_path)
            except Exception as e:
                print(f"[HistoryManager] Error cleaning up orphaned directories: {e}")

    def _save(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            print(f"[HistoryManager] Save error: {e}")

    def add_generation(self, model_id, image_path, state_path, renders_dict):
        with self.lock:
            # Check if model already exists, if so update it
            existing = next((m for m in self.models if m['id'] == model_id), None)
            
            # Find a representative thumbnail URL from renders_dict
            thumbnail_url = ""
            if renders_dict and 'shaded_forest' in renders_dict:
                first_render = renders_dict['shaded_forest'][0]
                # FileData object can be passed as dict or FileData
                if isinstance(first_render, dict) and 'path' in first_render:
                    filename = os.path.basename(first_render['path'])
                    thumbnail_url = f"/tmp/{model_id}/{filename}"
                elif hasattr(first_render, 'path'):
                    thumbnail_url = f"/tmp/{model_id}/{os.path.basename(first_render.path)}"
                elif isinstance(first_render, str):
                    thumbnail_url = f"/tmp/{model_id}/{os.path.basename(first_render)}"

            if not thumbnail_url and renders_dict:
                # Fallback to any render frame
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
            self.models.insert(0, entry)  # Most recent first
            self._save()

    def add_preprocess(self, model_id, image_path):
        with self.lock:
            existing = next((m for m in self.models if m['id'] == model_id), None)
            if existing and existing.get("status") in ("generated", "completed"):
                return # Keep higher status intact
            
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

    def add_glb(self, model_id, glb_path):
        with self.lock:
            existing = next((m for m in self.models if m['id'] == model_id), None)
            if not existing:
                # If generation wasn't registered somehow, create a skeleton entry
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
            
            # LRU Cleanup of extra models
            while len(self.models) > self.limit:
                oldest = self.models.pop()  # Pop oldest from the end
                self._delete_model_files(oldest)

            self._save()

    def _delete_model_files(self, model_entry):
        model_id = model_entry['id']
        print(f"[HistoryManager] Evicting model {model_id} via LRU limit...")
        job_dir = os.path.join(TMP_DIR, model_id)
        if os.path.exists(job_dir):
            import shutil
            try:
                shutil.rmtree(job_dir)
                print(f"[HistoryManager] Deleted job directory: {job_dir}")
            except Exception as e:
                print(f"[HistoryManager] Error deleting job directory: {e}")
                
        # Clean up old loose files directly in TMP_DIR older than 1 hour (3600 seconds)
        try:
            now = time.time()
            for f in os.listdir(TMP_DIR):
                f_path = os.path.join(TMP_DIR, f)
                if os.path.isfile(f_path) and (f_path.endswith('.png') or f_path.endswith('.json')) and (now - os.path.getmtime(f_path) > 3600):
                    os.remove(f_path)
        except Exception as e:
            print(f"[HistoryManager] Auto disk cleanup error: {e}")

    def get_all(self):
        with self.lock:
            return [m for m in self.models]

history_manager = HistoryManager(limit=50) # Server-side model cache limit: 50 models (approx 4500 render files, 90x model limit)

def resolve_state_path(state_path):
    # Resolve web/relative paths (e.g. "/tmp/model_123/state.npz") to absolute disk paths inside TMP_DIR
    if "/tmp/" in state_path or "\\tmp\\" in state_path:
        parts = state_path.replace("\\", "/").split("/tmp/")
        return os.path.join(TMP_DIR, parts[-1])
    elif "model_" in state_path and not state_path.startswith(TMP_DIR):
        parts = state_path.split("model_")
        return os.path.join(TMP_DIR, "model_" + parts[-1])
    return os.path.abspath(state_path)

def unpack_state(state_path):
    state_path = resolve_state_path(state_path)
    data = np.load(state_path)
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(data['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, int(data['res'])

# ============================================================================
# Progress Tracking (file-based, cross-process safe for @spaces.GPU)
# ============================================================================

import asyncio
from fastapi.responses import JSONResponse
from fastapi import Request

_thread_local = threading.local()

def _progress_file(model_id: str) -> str:
    """Return path to a model's progress JSON file inside its job directory."""
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
    """Atomically write progress JSON to a file (cross-process safe)."""
    path = _progress_file(model_id)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        pass

# Monkey-patch tqdm to intercept progress
import tqdm as _tqdm_module

_original_tqdm = _tqdm_module.tqdm

class _TqdmProgressInterceptor(_original_tqdm):
    """Wraps tqdm to push progress updates to SSE."""
    def __init__(self, *args, **kwargs):
        self._stage_desc = kwargs.get('desc', 'Processing')
        super().__init__(*args, **kwargs)
    
    def set_description(self, desc=None, refresh=True):
        self._stage_desc = desc or 'Processing'
        super().set_description(desc, refresh)
    
    def update(self, n=1):
        super().update(n)
        _update_progress(self._stage_desc, self.n, self.total or 0)

# Patch tqdm globally
_tqdm_module.tqdm = _TqdmProgressInterceptor
# Also patch the direct import in the sampler module and render_utils
import pixal3d.pipelines.samplers.flow_euler as _fe_module
_fe_module.tqdm = _TqdmProgressInterceptor
import pixal3d.utils.render_utils as _ru_module
_ru_module.tqdm = _TqdmProgressInterceptor
import o_voxel.postprocess as _ovp_module
_ovp_module.tqdm = _TqdmProgressInterceptor

# ============================================================================
# API Implementation
# ============================================================================

app = Server()

@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(
            content=f.read(),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )

@app.get("/sw.js")
async def service_worker():
    # Return empty / self-uninstalling service worker to fully clear client-side caches
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
    """Return server configuration for frontend (e.g. LOW_VRAM mode)."""
    return JSONResponse({"low_vram": LOW_VRAM})

@app.get("/progress")
async def progress_poll(request: Request):
    """Polling endpoint for real-time progress updates during generation."""
    model_id = request.query_params.get("model_id", "")
    path = _progress_file(model_id)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return JSONResponse(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return JSONResponse({"stage": "Waiting...", "step": 0, "total": 0, "done": False})

@app.api()
def preprocess(image: FileData, model_id: str) -> FileData:
    init_models()
    try:
        img = Image.open(image["path"])
        processed = pipeline.preprocess_image(img)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
            print("[OOM] CUDA Out of memory during preprocessing. Retrying on CPU...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # Force rembg to CPU
                pipeline.rembg_model.cpu()
                img = Image.open(image["path"])
                processed = pipeline.preprocess_image(img)
                # Restore rembg back to GPU if possible
                if torch.cuda.is_available():
                    pipeline.rembg_model.cuda()
            except Exception as cpu_err:
                print(f"[OOM] CPU fallback failed: {cpu_err}")
                raise gr.Error("CUDA Out of Memory during image preprocessing. Please try with a smaller image.")
        else:
            raise e
    except Exception as e:
        print(f"[Preprocess Error] {e}")
        raise gr.Error(f"Preprocessing failed: {str(e)}")
    
    # Store preprocess output directly in the job directory
    job_dir = os.path.join(TMP_DIR, model_id)
    os.makedirs(job_dir, exist_ok=True)
    out_path = os.path.join(job_dir, "image.png")
    processed.save(out_path)
    
    # Register preprocessing stage in the history manager so they can continue to generate
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
    
    if manual_fov > 0:
        # Convert to radians based on unit
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
            print("[OOM] CUDA Out of Memory during 3D generation. Clearing cache...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            raise gr.Error("CUDA Out of Memory: The GPU ran out of memory. Please select a lower resolution or retry when system load decreases.")
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
    
    # Save renders and return paths
    render_files = {}
    for mode_key, frames in renders.items():
        mode_files = []
        for i, frame in enumerate(frames):
            p = os.path.abspath(os.path.join(job_dir, f"render_{mode_key}_{i}.jpg"))
            Image.fromarray(frame).save(p, quality=85)
            mode_files.append(FileData(path=p))
        render_files[mode_key] = mode_files

    # Register this generation stage in the history manager
    history_manager.add_generation(model_id, image_copy_path, state_path, render_files)

    _finish_progress()
    return {
        "render_paths": render_files,
        "state_path": os.path.abspath(state_path),
        "camera_angle_x": camera_params['camera_angle_x'],
        "distance": camera_params['distance'],
    }

@app.api()
def extract_glb_api(state_path: str, decimation_target: int, texture_size: int, session_id: str = "") -> FileData:
    init_models()
    _reset_progress(session_id)
    _update_progress("Decoding latent", 0, 1)
    
    try:
        shape_slat, tex_slat, res = unpack_state(state_path)
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
        _update_progress("Decoding latent", 1, 1)
        
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target, texture_size=texture_size,
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
        )
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.OutOfMemoryError):
            print("[OOM] CUDA Out of Memory during GLB extraction. Clearing cache...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            raise gr.Error("CUDA Out of Memory: GLB extraction ran out of memory. Please lower the texture size and try again.")
        else:
            raise e
    except Exception as e:
        print(f"[GLB Extraction Error] {e}")
        raise gr.Error(f"GLB Extraction failed: {str(e)}")
        
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)
    
    # Extract model_id and job directory from state path
    resolved_state_path = resolve_state_path(state_path)
    job_dir = os.path.dirname(resolved_state_path)
    model_id = os.path.basename(job_dir)
        
    out_glb = os.path.join(job_dir, "result.glb")
    glb.export(out_glb, extension_webp=True)
    
    # Add extracted GLB to history manager (which enforces LRU limit and deletes extra files)
    history_manager.add_glb(model_id, out_glb)
    
    _finish_progress()
    return FileData(path=out_glb)

@app.get("/history")
async def get_history():
    """Return the list of old generated 3D models."""
    return JSONResponse(history_manager.get_all())

# Mount assets and tmp with explicit .glb MIME-type mappings
import mimetypes
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")

# Add middleware for robust CORS and caching policies
from fastapi.middleware.cors import CORSMiddleware
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
    import sys
    parser = argparse.ArgumentParser(description="Pixal3D Demo Server")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models lazy-load to GPU per stage.")
    args, remaining = parser.parse_known_args()
    if args.low_vram:
        LOW_VRAM = True

    # Generate optimized thumbnails for speedy gallery loading
    build_thumbnails()
    
    # Pre-initialize models before launching the server
    init_models()
    
    app.launch(show_error=True, share=True)