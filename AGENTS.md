# Agent Guide: Pixal3D

This guide is designed to help future AI agents quickly understand, navigate, and work productively within the **Pixal3D** repository. It focuses on architecture, workflow pipelines, key configuration details, coordinate systems, and non-obvious engineering gotchas that save trial-and-error discovery.

---

## 🗺️ Repository Architecture & Design Patterns

Pixal3D is a state-of-the-art pixel-aligned 3D asset generator accepted at **SIGGRAPH 2026**. It generates high-fidelity, view-aligned meshes with PBR materials from a single image.

Unlike standard approaches that inject conditioning features loosely via cross-attention, Pixal3D explicitly **projects and lifts 2D features into 3D space** (established by a DinoV3 projection feature extractor) to enforce direct pixel-to-3D alignments.

### 1. Model Structure & Cascades
Generation works in a **three-stage cascade**, progressively upsampling resolutions to build geometry and materials:

1. **Stage 1: Sparse Structure (SS)** (`32` → `64` voxel resolution): Predicts the basic spatial occupancy grids using a sparse convolutional/transformer backbone.
2. **Stage 2: Shape (SLat Shape)** (`256` → `512` → `1024` resolution): Models structured latents representing continuous surface details.
3. **Stage 3: Texture (SLat PBR/Tex)** (`256` → `512` → `1024` resolution): Predicts the PBR material attributes (base color, metalness, roughness, and alpha) aligned over the generated shape.

### 2. Core Directory Layout
*   `pixal3d/`: Core package containing models, representation backbones, datasets, trainers, and pipeline definitions.
    *   `models/`: Sparse structures, Sparse UNet VAEs, Sparse Structure/Structured Latent Flow models (backbone definitions).
    *   `representations/`: 3D voxel representations and mesh processing/filling utilities.
    *   `datasets/`: Data loading components (e.g., `flexi_dual_grid`, `structured_latent`, etc.).
    *   `trainers/`: Flow-matching objectives, VAE trainers, and projection mixins (`image_conditioned_proj.py`).
    *   `pipelines/`: Generation inference pipelines, camera/sampling controllers, and background removal tools (`rembg`).
*   `data_toolkit/`: Scripts and recipes to download, process, voxelize, and encode multi-view 3D assets into view-aligned latents.
*   `configs/`: Hyperparameters and JSON configurations for the training stages.
*   `inference.py` / `app.py`: Standard entrypoints for inference serving (CLI and Gradio/FastAPI web interface).
*   `train.py`: Multiprocessing distributed trainer interface.

---

## 🛠️ Environment & Setup Context

The project is heavily based on the **TRELLIS.2** backbone. To set up the environment:
1. Initialize TRELLIS.2 base dependencies.
2. Run `pip install -r requirements.txt`.
3. Build **natten** matching your CUDA architecture:
   ```bash
   NATTEN_CUDA_ARCH="xx" NATTEN_N_WORKERS=xx pip install natten==0.21.0 --no-build-isolation
   ```
4. Install custom **utils3d** wheel:
   ```bash
   pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
   ```

---

## ⚡ Execution and Commands Reference

### 1. Inference CLI
Run generation from a single input image to standard GLB:
```bash
python inference.py --image assets/images/0_img.png --output ./output.glb
```

*   **Low-VRAM mode**: Cuts GPU memory footprint in half (to ~10-12 GB) by keeping flow models/extractors on the CPU and loading them to the GPU on-demand during their active stage.
    ```bash
    python inference.py --image assets/images/0_img.png --output ./output.glb --low_vram
    ```
*   **Resolution Override**: Forces specific upsample limits.
    ```bash
    python inference.py --image assets/images/0_img.png --output ./output.glb --resolution 1536
    ```
*   **Manual FOV Override**: If estimated camera projection distorts output, manually specify FOV in radians:
    ```bash
    python inference.py --image assets/images/0_img.png --output ./output.glb --fov 0.2
    ```

### 2. Gradio Web Application
Launch interactive GUI serving a custom FastAPI frontend:
```bash
python app.py [--low_vram]
```

### 3. Data Preparation Toolkit
The toolkit is in `data_toolkit/`. Follow these sequential scripts to prepare training data:
```bash
# Step 1: Install data-prep dependencies
. ./data_toolkit/setup.sh

# Step 2: Initialize metadata registry
python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab --root datasets/ObjaverseXL_sketchfab

# Step 3: Download assets (supports multi-node rank/world_size)
python data_toolkit/download.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --world_size 160000

# Step 4: Extract standardized meshes and PBR materials (CPU-bound)
python data_toolkit/dump_mesh.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/dump_pbr.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/asset_stats.py --root datasets/ObjaverseXL_sketchfab

# Step 5: Render image conditions (automatically initializes Blender)
python data_toolkit/render_cond.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab

# Step 6: Voxelize meshes and PBR maps into view-aligned O-Voxels
python data_toolkit/dual_grid_view.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256 --view_indices 0-1
python data_toolkit/voxelize_pbr_view.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256 --view_indices 0-1

# Step 7: Encode latents for Flow Model training
python data_toolkit/encode_shape_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 1024 --view_indices 0-1
python data_toolkit/encode_pbr_latent_view.py --root datasets/ObjaverseXL_sketchfab --resolution 1024 --view_indices 0-1
# Note: Update metadata registry BEFORE generating Sparse Structure (SS) latents
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/encode_ss_latent_view.py --root datasets/ObjaverseXL_sketchfab --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024_view --resolution 64 --view_indices 0-1
```

### 4. Training Pipeline
Each stage of the cascade requires executing `train.py` with the appropriate resolution-specific JSON configuration:
```bash
python train.py \
  --config configs/gen/<STAGE_CONFIG>.json \
  --output_dir <OUTPUT_DIR> \
  --data_dir '<DATA_DIR_JSON>'
```
*   `--data_dir` must be passed as a JSON layout config:
    `'{"ObjaverseXL_sketchfab": {"base": "datasets/ObjaverseXL_sketchfab", "shape_latent": "...", "render_cond": "..."}}'`

---

## 🧠 Critical Gotchas & Coordination Space Details

### 1. 3D Coordinate Mapping (Crucial for Projection)
The pixel-aligned lifting relies on extreme math precision. Keep this mental model of the spaces in mind:
*   **Mesh Space**: Mesh vertices are defined in the normalized cube `[-0.5, 0.5]^3`.
*   **Voxel Grid / ProjGrid Space**: Grid points occupy `[-1, 1]^3` (constructed from `torch.linspace(-1, 1, res)`).
*   **Coordinate Transformations**: To map grid points into Blender projection camera coordinates:
    1. A Y-Z axis swap rotation matrix is applied: $x' = x$, $y' = -z$, $z' = y$.
    2. Grid points are scaled down using: $\text{points} / \text{mesh\_scale} / 2$.
    3. Project onto the image space using the `front_view_transform_matrix` and estimated camera `distance`.
*   **Verts Scaling Alignment**: In standard dataset visualization (`structured_latent_shape.py`), mesh vertices must be scaled by `/ mesh_scale` to fit within ProjGrid's expected projection coordinates.

### 2. VRAM Optimizations & On-Demand Lifecycle
*   During wild image inference, **MoGe-2** (`Ruicheng/moge-2-vitl`) is loaded to estimate camera properties. This is a heavy model.
*   **Constraint**: Once camera estimation completes, MoGe-2 **must be moved back to CPU and deleted** (`moge_model.cpu(); del moge_model; torch.cuda.empty_cache()`) before pipeline initialization. Failure to do so will cause immediate Out-Of-Memory (OOM) errors during the subsequent cascade generation stages.

### 3. W&B logging & S3 FUSE issues
*   When training on cloud environments with S3 FUSE directories (like cloud storage mounted as a local directory), `wandb` metadata updates can freeze or crash due to direct rename/append limitations on FUSE mounts.
*   **Workaround**: Ensure the `WANDB_DIR` environment variable is explicitly configured to point to a fast, local SSD storage space (e.g., `/tmp` or direct instance storage) rather than letting it fall back to the remote `output_dir` mount.

### 4. OpenCV EXR Handling
*   The pipeline relies heavily on high-dynamic range `.exr` environment map files (located in `assets/hdri/`).
*   **Gotcha**: Python OpenCV does not enable OpenEXR support by default. You **must** set the environment flag prior to loading any images:
    ```python
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    ```

### 5. FlexGEMM Autotuning
*   Pixal3D uses sparse operations and autotuned GEMM kernels. The path to the cache is explicitly declared via:
    ```python
    os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
    ```
*   Ensure this location is writeable by the running agent/process.

### 6. Attention Backend Fallback
*   If `flash_attn` is missing on your host machine, the built-in PyTorch Scaled Dot Product Attention (SDPA) backend can be forced by setting:
    ```python
    os.environ["ATTN_BACKEND"] = "sdpa"
    ```

### 7. Metadata Rebuilding and Index Deduplication
*   `build_metadata.py` scans downloaded records and updates central index registries.
*   **Gotcha**: If multi-view generation creates duplicate sha256 entries in temporary pandas records, index deduplication must occur via `.groupby(level=0).first()` before invoking `.combine_first()`. Otherwise, panda operations will raise alignment errors.
