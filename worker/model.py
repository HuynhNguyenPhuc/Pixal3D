"""Model worker for Pixal3D generation tasks."""

import os
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

import gc
import time
import uuid

from PIL import Image

import torch
import trimesh

import o_voxel
import numpy as np

import config
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from inference import get_camera_params_wild_moge
from utilities.image import convert_to_pil_image
from utilities.logger import get_logger
from utilities.gpu import aggressive_gpu_cleanup, clean_mesh_vertices_faces


# Minimal mocks for IMAGE_COND_CONFIGS
IMAGE_COND_CONFIGS = {
    'ss': {'model_name': 'camenduru/dinov3-vitl16-pretrain-lvd1689m', 'image_size': 512, 'grid_resolution': 16},
    'shape_512': {'model_name': 'camenduru/dinov3-vitl16-pretrain-lvd1689m', 'image_size': 512, 'grid_resolution': 32, 'use_naf_upsample': True, 'naf_target_size': 512},
    'shape_1024': {'model_name': 'camenduru/dinov3-vitl16-pretrain-lvd1689m', 'image_size': 1024, 'grid_resolution': 64, 'use_naf_upsample': True, 'naf_target_size': 512},
    'tex_1024': {'model_name': 'camenduru/dinov3-vitl16-pretrain-lvd1689m', 'image_size': 1024, 'grid_resolution': 64, 'use_naf_upsample': True, 'naf_target_size': 1024},
}

WILD_MESH_SCALE = 1.0
WILD_EXTEND_PIXEL = 0
WILD_IMAGE_RESOLUTION = 512


def build_image_cond_model(config):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model

def load_moge_model(device='cuda', model_name='Ruicheng/moge-2-vitl'):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model


class ModelWorker:
    def __init__(self, model_path='TencentARC/Pixal3D', device='cuda', worker_id=None, save_dir='gradio_cache'):
        self.model_path = model_path
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.save_dir = save_dir
        self.request_count = 0

        self.logger = get_logger(self.worker_id)
        self.logger.info(f'Loading model {model_path} on worker {self.worker_id}')

        torch.backends.cudnn.benchmark = True

        self.pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
        
        self.logger.info('Building DinoV3ProjFeatureExtractor models...')
        self.pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS['ss'])
        self.pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS['shape_512'])
        self.pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS['shape_1024'])
        self.pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS['tex_1024'])
        
        self.pipeline.low_vram = False
        self.pipeline.cuda()
        self.pipeline.image_cond_model_ss.cuda()
        self.pipeline.image_cond_model_shape_512.cuda()
        self.pipeline.image_cond_model_shape_1024.cuda()
        self.pipeline.image_cond_model_tex_1024.cuda()
        
        self.logger.info('Pre-loading NAF upsampler model...')
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(self.pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()

        self.logger.info('Loading MoGe-2 for camera estimation (CPU)...')
        self.moge_model = load_moge_model(device='cpu')

        self.logger.info('Worker initialization complete.')
        aggressive_gpu_cleanup()

    def generate(self, uid: str, params: dict) -> str:
        image_payload = params.get("image")
        seed = params.get("seed", 42)
        mode = params.get("mode", "staged")
        
        # Hardcode resolution to 1536 for best quality as requested
        resolution = 1536
        decimation_target = params.get("decimation_target", 500000)
        texture_size = params.get("texture_size", 4096)
        no_webp_val = params.get("no_webp", True)
        if isinstance(no_webp_val, str):
            no_webp = no_webp_val.lower() in ("true", "1")
        else:
            no_webp = bool(no_webp_val)
        
        ss_sampler_params = {
            'steps': params.get("ss_sampling_steps", 12),
            'guidance_strength': params.get("ss_guidance_strength", 7.5),
            'guidance_rescale': params.get("ss_guidance_rescale", 0.7),
            'rescale_t': params.get("ss_rescale_t", 5.0)
        }
        shape_sampler_params = {
            'steps': params.get("shape_slat_sampling_steps", 12),
            'guidance_strength': params.get("shape_slat_guidance_strength", 7.5),
            'guidance_rescale': params.get("shape_slat_guidance_rescale", 0.5),
            'rescale_t': params.get("shape_slat_rescale_t", 3.0)
        }
        tex_sampler_params = {
            'steps': params.get("tex_slat_sampling_steps", 12),
            'guidance_strength': params.get("tex_slat_guidance_strength", 1.0),
            'guidance_rescale': params.get("tex_slat_guidance_rescale", 0.0),
            'rescale_t': params.get("tex_slat_rescale_t", 3.0)
        }
        
        pipeline_type = f"{resolution}_cascade"

        self.logger.info(f'Task {uid}: processing {mode} with resolution {resolution}')
        start_time = time.time()
        self.request_count += 1
        
        if self.request_count % 10 == 0:
            torch.cuda.empty_cache()

        image = convert_to_pil_image(image_payload)
        image = self.pipeline.preprocess_image(image)
        
        temp_path = os.path.join(self.save_dir, f'{uid}_temp.png')
        image.save(temp_path)
        
        fov = params.get("fov", -1.0)
        if fov > 0:
            camera_angle_x = float(fov)
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            from inference import distance_from_fov
            distance = distance_from_fov(
                camera_angle_x, grid_point,
                torch.tensor([0 - WILD_EXTEND_PIXEL, WILD_IMAGE_RESOLUTION - 1 + WILD_EXTEND_PIXEL]),
                WILD_MESH_SCALE, WILD_IMAGE_RESOLUTION
            )["distance_from_x"]
            camera_params = {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': WILD_MESH_SCALE}
            self.logger.info(f"Using manual FOV: {camera_angle_x:.4f} rad, distance: {distance:.4f}")
        else:
            # Load moge to GPU temporarily
            self.moge_model.to('cuda')
            camera_params = get_camera_params_wild_moge(
                temp_path, device='cuda',
                mesh_scale=WILD_MESH_SCALE, extend_pixel=WILD_EXTEND_PIXEL,
                image_resolution=WILD_IMAGE_RESOLUTION,
                moge_model=self.moge_model
            )
            # Unload back to CPU to save memory for pipeline
            self.moge_model.cpu()
            
        os.remove(temp_path)

        mesh_list, (shape_slat, tex_slat, res) = self.pipeline.run(
            image,
            camera_params=camera_params,
            seed=seed,
            preprocess_image=False,
            sparse_structure_sampler_params=ss_sampler_params,
            shape_slat_sampler_params=shape_sampler_params,
            tex_slat_sampler_params=tex_sampler_params,
            return_latent=True,
            pipeline_type=pipeline_type,
        )
        
        mesh = mesh_list[0]

        # Free heavy sparse latents and other intermediate structures that are no longer needed for triangulation to save massive GPU VRAM.
        del shape_slat, tex_slat, mesh_list
        aggressive_gpu_cleanup()

        # Clean vertices and faces to prevent cumesh illegal memory access from degenerate geometry
        self.logger.info(f"Cleaning mesh vertices/faces before exporting GLB (input: {mesh.vertices.shape[0]:,} vertices, {mesh.faces.shape[0]:,} faces)...")
        mesh_vertices, mesh_faces = clean_mesh_vertices_faces(mesh.vertices, mesh.faces)
        self.logger.info(f"Mesh cleaned (output: {mesh_vertices.shape[0]:,} vertices, {mesh_faces.shape[0]:,} faces)")

        # Proactive CPU Simplification Safeguard
        # If the cleaned mesh has >= SIMPLIFICATION_THRESHOLD_FACES faces, reduce it to SIMPLIFICATION_TARGET_FACES faces on CPU to prevent GPU OOM while keeping the reconstruction grid_size at exactly 1536.
        if mesh_faces.shape[0] >= config.SIMPLIFICATION_THRESHOLD_FACES:
            self.logger.info(f"Proactive Safeguard: Mesh has >= {config.SIMPLIFICATION_THRESHOLD_FACES:,} faces ({mesh_faces.shape[0]:,} faces). Simplifying to {config.SIMPLIFICATION_TARGET_FACES:,} faces on CPU to prevent GPU OOM...")
            try:
                device = mesh_vertices.device
                tm = trimesh.Trimesh(vertices=mesh_vertices.cpu().numpy(), faces=mesh_faces.cpu().numpy(), process=False)
                tm = tm.simplify_quadric_decimation(config.SIMPLIFICATION_TARGET_FACES)
                mesh_vertices = torch.from_numpy(tm.vertices).float().to(device)
                mesh_faces = torch.from_numpy(tm.faces).int().to(device)
                self.logger.info(f"Proactive CPU Simplification complete. New face count: {mesh_faces.shape[0]:,}")
            except Exception as e:
                self.logger.warning(f"Proactive CPU Simplification failed: {e}. Proceeding with original density.")

        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_vertices, faces=mesh_faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=self.pipeline.pbr_attr_layout,
                grid_size=res, 
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target, texture_size=texture_size,
                remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False,
            )
        except RuntimeError as exc:
            exc_str = str(exc).lower()
            if (("cuda error" in exc_str or "cumesh" in exc_str) and "out of memory" not in exc_str) or "illegal memory access" in exc_str or "device-side assertion" in exc_str:
                self.logger.error(f"Fatal CUDA/CuMesh error: {exc_str}. Marking for restart.")
                import app_state
                app_state.needs_subprocess_restart = True
                raise exc
            if "out of memory" in exc_str or "cuda error" in exc_str:
                self.logger.warning(
                    f"OOM during to_glb remesh for uid={uid} "
                    f"({mesh_faces.shape[0]:,} faces); "
                    f"retrying without remesh"
                )

                # Clean up GPU memory before retrying the export.
                aggressive_gpu_cleanup()

                try:
                    glb = o_voxel.postprocess.to_glb(
                        vertices=mesh_vertices, faces=mesh_faces, attr_volume=mesh.attrs,
                        coords=mesh.coords, attr_layout=self.pipeline.pbr_attr_layout,
                        grid_size=res, 
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=decimation_target, texture_size=texture_size,
                        remesh=False, remesh_band=1, remesh_project=0, use_tqdm=False,
                    )
                except RuntimeError as exc_fallback:
                    exc_fallback_str = str(exc_fallback).lower()
                    if "out of memory" in exc_fallback_str or "cuda error" in exc_fallback_str:
                        self.logger.warning(f"OOM during to_glb no-remesh fallback for uid={uid}. Attempting CPU-decimation fallback and lowering texture size to prevent worker crash.")
                        aggressive_gpu_cleanup()
                        
                        import trimesh
                        device = mesh_vertices.device
                        tm = trimesh.Trimesh(vertices=mesh_vertices.cpu().numpy(), faces=mesh_faces.cpu().numpy(), process=False)
                        cpu_target = max(decimation_target, 200000)
                        self.logger.info(f"Simplifying mesh on CPU to {cpu_target:,} faces...")
                        tm = tm.simplify_quadric_decimation(cpu_target)
                        
                        mesh_vertices_fallback = torch.from_numpy(tm.vertices).float().to(device)
                        mesh_faces_fallback = torch.from_numpy(tm.faces).int().to(device)
                        
                        self.logger.info("Retrying to_glb on CPU-simplified mesh (no remesh, 1024 texture)...")
                        glb = o_voxel.postprocess.to_glb(
                            vertices=mesh_vertices_fallback, faces=mesh_faces_fallback, attr_volume=mesh.attrs,
                            coords=mesh.coords, attr_layout=self.pipeline.pbr_attr_layout,
                            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=decimation_target, texture_size=min(texture_size, 1024),
                            remesh=False, remesh_band=1, remesh_project=0, use_tqdm=False,
                        )
                    else:
                        raise exc_fallback
        
        rot = np.array([
            [ 1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1],
        ], dtype=np.float64)
        glb.apply_transform(rot)

        output_path = os.path.join(self.save_dir, f'{uid}.glb')
        glb.export(output_path, extension_webp=not no_webp)

        end_time = time.time()
        self.logger.info(f'Task {uid}: generation completed in {end_time - start_time:.2f} seconds.')
        
        aggressive_gpu_cleanup()

        return output_path
