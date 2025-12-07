"""
CUDA Worker Process for Text-to-Image-to-3D pipeline.
Runs in a separate process to avoid Gradio 6 asyncio interference with CUDA.
Supports multiple 3D engines: TripoSR and Hunyuan3D-2.
"""
import io
import logging
import multiprocessing as mp
import tempfile
import time
from typing import Optional

import numpy as np
from PIL import Image

from .config import settings, CHECKPOINTS

# Global queues for IPC
task_queue: Optional[mp.Queue] = None
result_queue: Optional[mp.Queue] = None
worker_process: Optional[mp.Process] = None

# Logger
logger = logging.getLogger(__name__)


def cuda_worker_process(task_q: mp.Queue, result_q: mp.Queue):
    """
    CUDA worker process that handles all GPU operations.
    Runs in a separate process to avoid asyncio interference.
    """
    import torch
    import rembg
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # Setup logging for worker
    logging.basicConfig(level=logging.INFO, format="[Worker] %(message)s")
    worker_logger = logging.getLogger("worker")

    device = settings.device if settings.use_gpu and torch.cuda.is_available() else "cpu"
    loaded_checkpoint = None

    # Model instances
    triposr_model = None
    hunyuan_shapegen = None
    hunyuan_texgen = None
    hunyuan_rembg = None
    hunyuan_mv_shapegen = None  # Multi-view model

    worker_logger.info(f"Initializing models on {device}...")

    try:
        # Initialize SDXL-Lightning
        pipe = StableDiffusionXLPipeline.from_pretrained(
            settings.base_model,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            variant="fp16" if "cuda" in device else None,
        ).to(device)

        # Initialize rembg
        rembg_session = rembg.new_session()

        # Initialize TripoSR (lazy load on first use)
        def get_triposr():
            nonlocal triposr_model
            if triposr_model is None:
                worker_logger.info("Loading TripoSR model...")
                from tsr.system import TSR
                triposr_model = TSR.from_pretrained(
                    settings.triposr_model,
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                triposr_model.renderer.set_chunk_size(settings.chunk_size)
                triposr_model.to(device)
                worker_logger.info("TripoSR loaded!")
            return triposr_model

        # Initialize Hunyuan3D-2 (lazy load on first use)
        def get_hunyuan():
            nonlocal hunyuan_shapegen, hunyuan_texgen, hunyuan_rembg
            if hunyuan_shapegen is None:
                worker_logger.info("Loading Hunyuan3D-2 models...")
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
                from hy3dgen.rembg import BackgroundRemover

                hunyuan_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    settings.hunyuan3d_model
                )
                hunyuan_rembg = BackgroundRemover()

                # Try to load texture generator (requires custom_rasterizer)
                try:
                    from hy3dgen.texgen import Hunyuan3DPaintPipeline
                    hunyuan_texgen = Hunyuan3DPaintPipeline.from_pretrained(
                        settings.hunyuan3d_model
                    )
                    worker_logger.info("Hunyuan3D-2 texture generator loaded!")
                except Exception as e:
                    worker_logger.warning(f"Texture generator not available: {e}")
                    hunyuan_texgen = None

                worker_logger.info("Hunyuan3D-2 shape generator loaded!")
            return hunyuan_shapegen, hunyuan_texgen, hunyuan_rembg

        # Initialize Hunyuan3D-2 MV (Multi-View) model
        def get_hunyuan_mv():
            nonlocal hunyuan_mv_shapegen, hunyuan_texgen, hunyuan_rembg
            if hunyuan_mv_shapegen is None:
                worker_logger.info("Loading Hunyuan3D-2 MV (Multi-View) model...")
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
                from hy3dgen.rembg import BackgroundRemover

                hunyuan_mv_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    settings.hunyuan3d_mv_model,
                    subfolder='hunyuan3d-dit-v2-mv'
                )
                if hunyuan_rembg is None:
                    hunyuan_rembg = BackgroundRemover()

                # Try to load texture generator (shared with single-view)
                if hunyuan_texgen is None:
                    try:
                        from hy3dgen.texgen import Hunyuan3DPaintPipeline
                        hunyuan_texgen = Hunyuan3DPaintPipeline.from_pretrained(
                            settings.hunyuan3d_model
                        )
                        worker_logger.info("Hunyuan3D-2 texture generator loaded!")
                    except Exception as e:
                        worker_logger.warning(f"Texture generator not available: {e}")

                worker_logger.info("Hunyuan3D-2 MV model loaded!")
            return hunyuan_mv_shapegen, hunyuan_texgen, hunyuan_rembg

        def unload_hunyuan():
            """Unload Hunyuan3D-2 models to free GPU memory."""
            nonlocal hunyuan_shapegen, hunyuan_texgen, hunyuan_rembg, hunyuan_mv_shapegen
            if hunyuan_shapegen is not None or hunyuan_texgen is not None or hunyuan_mv_shapegen is not None:
                worker_logger.info("[Memory] Unloading Hunyuan3D-2 models...")

                # Delete model references
                if hunyuan_texgen is not None:
                    del hunyuan_texgen
                    hunyuan_texgen = None
                if hunyuan_shapegen is not None:
                    del hunyuan_shapegen
                    hunyuan_shapegen = None
                if hunyuan_mv_shapegen is not None:
                    del hunyuan_mv_shapegen
                    hunyuan_mv_shapegen = None
                if hunyuan_rembg is not None:
                    del hunyuan_rembg
                    hunyuan_rembg = None

                # Force garbage collection and clear CUDA cache
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                # Log memory status
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    worker_logger.info(f"[Memory] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        worker_logger.info("Base models initialized successfully!")
        result_q.put({"status": "READY"})

    except Exception as e:
        worker_logger.error(f"Failed to initialize models: {e}")
        result_q.put({"status": "ERROR", "error": str(e)})
        return

    def fill_background(image: Image.Image, color: tuple = (127, 127, 127)) -> Image.Image:
        """Fill transparent background with specified color."""
        if image.mode != "RGBA":
            return image.convert("RGB")
        img_array = np.array(image).astype(np.float32) / 255.0
        bg = np.array(color) / 255.0
        rgb = img_array[:, :, :3] * img_array[:, :, 3:4] + bg * (1 - img_array[:, :, 3:4])
        return Image.fromarray((rgb * 255.0).astype(np.uint8))

    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        import base64
        buf = io.BytesIO()
        image.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def process_3d_triposr(input_image: Image.Image, mc_resolution: int) -> tuple:
        """Generate 3D using TripoSR."""
        from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
        from tsr.bake_texture import bake_texture
        import trimesh

        model = get_triposr()

        # Generate 3D
        worker_logger.info("[TripoSR] Generating 3D model...")
        t0 = time.time()
        scene_codes = model(input_image, device=device)
        worker_logger.info(f"[TripoSR] Scene encoding: {time.time() - t0:.2f}s")

        t0 = time.time()
        mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
        worker_logger.info(f"[TripoSR] Mesh extraction: {time.time() - t0:.2f}s")

        mesh = to_gradio_3d_orientation(mesh)

        # Export OBJ with vertex colors
        obj_file = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        mesh.export(obj_file.name)

        # Export GLB with baked texture
        glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        try:
            worker_logger.info("[TripoSR] Baking texture...")
            texture_data = bake_texture(mesh, model, scene_codes[0], settings.default_texture_resolution)
            new_vertices = mesh.vertices[texture_data["vmapping"]]
            new_faces = texture_data["indices"]
            new_uvs = texture_data["uvs"]
            texture_colors = texture_data["colors"]
            texture_image = Image.fromarray(
                (np.clip(texture_colors, 0, 1) * 255).astype(np.uint8)
            )
            texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_image,
                metallicFactor=0.0,
                roughnessFactor=0.5,
            )
            uv_visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
            textured_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, visual=uv_visual)
            textured_mesh.export(glb_file.name)
        except Exception as tex_err:
            worker_logger.warning(f"[TripoSR] Texture baking failed: {tex_err}")
            mesh.export(glb_file.name)

        return obj_file.name, glb_file.name

    # Mesh quality settings (max faces for UV wrapping)
    MESH_QUALITY_MAP = {
        "fast": 100000,      # ~1-2 min texture generation
        "balanced": 200000,  # ~2-4 min texture generation
        "high": None,        # No limit, can take 5-10+ min
    }

    def process_3d_hunyuan(input_image: Image.Image, with_texture: bool = True, mesh_quality: str = "balanced") -> tuple:
        """Generate 3D using Hunyuan3D-2."""
        shapegen, texgen, hy_rembg = get_hunyuan()
        max_faces = MESH_QUALITY_MAP.get(mesh_quality, 200000)
        worker_logger.info(f"[Hunyuan3D] Mesh quality: {mesh_quality} (max_faces={max_faces})")

        # Ensure RGBA for Hunyuan
        if input_image.mode != "RGBA":
            worker_logger.info("[Hunyuan3D] Removing background...")
            input_image = hy_rembg(input_image.convert("RGB"))

        # Generate shape
        worker_logger.info("[Hunyuan3D] Generating 3D shape...")
        t0 = time.time()
        mesh = shapegen(image=input_image)[0]
        worker_logger.info(f"[Hunyuan3D] Shape generation: {time.time() - t0:.2f}s")

        # Generate texture if available and requested
        if with_texture and texgen is not None:
            try:
                worker_logger.info("[Hunyuan3D] Generating texture...")
                t0 = time.time()
                mesh = texgen(mesh, image=input_image, max_faces=max_faces)
                worker_logger.info(f"[Hunyuan3D] Texture generation: {time.time() - t0:.2f}s")
            except Exception as tex_err:
                worker_logger.warning(f"[Hunyuan3D] Texture generation failed: {tex_err}")
        elif with_texture:
            worker_logger.info("[Hunyuan3D] Texture generator not available, exporting shape only")

        # Export mesh
        obj_file = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        mesh.export(obj_file.name)
        mesh.export(glb_file.name)

        # Unload Hunyuan3D models to free GPU memory for next SDXL generation
        unload_hunyuan()

        return obj_file.name, glb_file.name

    def process_3d_hunyuan_mv(images: dict, with_texture: bool = True, mesh_quality: str = "balanced") -> tuple:
        """Generate 3D using Hunyuan3D-2 MV (Multi-View).

        Args:
            images: Dictionary with 'front', 'left', 'back' keys containing PIL Images
            with_texture: Whether to generate texture
            mesh_quality: Quality setting for mesh
        """
        mv_shapegen, texgen, hy_rembg = get_hunyuan_mv()
        max_faces = MESH_QUALITY_MAP.get(mesh_quality, 200000)
        worker_logger.info(f"[Hunyuan3D-MV] Mesh quality: {mesh_quality} (max_faces={max_faces})")

        # Prepare images - ensure RGBA for all views
        prepared_images = {}
        for view_name, img in images.items():
            if img is None:
                continue
            if img.mode != "RGBA":
                worker_logger.info(f"[Hunyuan3D-MV] Removing background from {view_name} view...")
                img = hy_rembg(img.convert("RGB"))
            prepared_images[view_name] = img

        if "front" not in prepared_images:
            raise Exception("Front view image is required for multi-view generation")

        worker_logger.info(f"[Hunyuan3D-MV] Using views: {list(prepared_images.keys())}")

        # Generate shape with multi-view
        worker_logger.info("[Hunyuan3D-MV] Generating 3D shape from multi-view images...")
        t0 = time.time()
        mesh = mv_shapegen(image=prepared_images)[0]
        worker_logger.info(f"[Hunyuan3D-MV] Shape generation: {time.time() - t0:.2f}s")

        # Generate texture if available and requested (use front image for texture)
        if with_texture and texgen is not None:
            try:
                worker_logger.info("[Hunyuan3D-MV] Generating texture...")
                t0 = time.time()
                mesh = texgen(mesh, image=prepared_images["front"], max_faces=max_faces)
                worker_logger.info(f"[Hunyuan3D-MV] Texture generation: {time.time() - t0:.2f}s")
            except Exception as tex_err:
                worker_logger.warning(f"[Hunyuan3D-MV] Texture generation failed: {tex_err}")
        elif with_texture:
            worker_logger.info("[Hunyuan3D-MV] Texture generator not available, exporting shape only")

        # Export mesh
        obj_file = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        mesh.export(obj_file.name)
        mesh.export(glb_file.name)

        # Unload models
        unload_hunyuan()

        return obj_file.name, glb_file.name

    def process_3d_hunyuan_api(input_image: Image.Image, mesh_quality: str = "balanced") -> tuple:
        """Generate 3D using Hunyuan Cloud API."""
        from .external_apis import get_hunyuan_client

        client = get_hunyuan_client()
        if client is None:
            raise Exception(
                "Hunyuan API credentials not configured. "
                "Set T2I3D_HUNYUAN_SECRET_ID and T2I3D_HUNYUAN_SECRET_KEY environment variables."
            )

        return client.image_to_3d(input_image, quality=mesh_quality)

    def generate_multiview_with_gemini(input_image: Image.Image) -> dict:
        """Generate multi-view images using Gemini API (Nano Banana Pro).

        Args:
            input_image: Input PIL Image (front view)

        Returns:
            Dictionary with 'front', 'left', 'back' keys containing PIL Images
        """
        from .external_apis import get_gemini_client

        client = get_gemini_client()
        if client is None:
            raise Exception(
                "Gemini API key not configured. "
                "Set T2I3D_GEMINI_API_KEY environment variable."
            )

        worker_logger.info("[Gemini] Generating multi-view images...")
        t0 = time.time()
        views = client.generate_multiview(input_image, views=["front", "left", "right", "back"])
        worker_logger.info(f"[Gemini] Multi-view generation: {time.time() - t0:.2f}s")

        # Check if we got all views
        generated_views = list(views.keys())
        worker_logger.info(f"[Gemini] Generated views: {generated_views}")

        return views

    def process_3d_gemini_mv(input_image: Image.Image, mesh_quality: str = "balanced") -> tuple:
        """Generate 3D using Gemini multi-view + Hunyuan3D-2 MV pipeline.

        Workflow:
        1. Single image → Gemini (multi-view generation)
        2. Multi-view images → Hunyuan3D-2 MV (3D generation)

        Args:
            input_image: Input PIL Image
            mesh_quality: Quality setting for mesh

        Returns:
            Tuple of (obj_path, glb_path, mv_images)
            mv_images is a dict with 'front', 'left', 'back' keys
        """
        worker_logger.info("[Gemini-MV] Starting Gemini multi-view → Hunyuan3D-2 MV pipeline...")
        total_start = time.time()

        # Step 1: Generate multi-view images with Gemini
        mv_images = generate_multiview_with_gemini(input_image)

        # Step 2: Generate 3D with Hunyuan3D-2 MV
        obj_path, glb_path = process_3d_hunyuan_mv(mv_images, mesh_quality=mesh_quality)

        worker_logger.info(f"[Gemini-MV] Total pipeline: {time.time() - total_start:.2f}s")
        return obj_path, glb_path, mv_images

    def process_text_to_3d(task: dict) -> dict:
        """Process full text-to-3D pipeline."""
        nonlocal loaded_checkpoint

        from .translator import translate_prompt
        from tsr.utils import remove_background, resize_foreground

        prompt = task["prompt"]
        prompt = translate_prompt(prompt)
        checkpoint_name = task["checkpoint"]
        do_remove_background = task["remove_background"]
        foreground_ratio = task["foreground_ratio"]
        mc_resolution = task["mc_resolution"]
        engine_3d = task.get("engine_3d", settings.default_3d_engine)
        mesh_quality = task.get("mesh_quality", "balanced")

        total_start = time.time()

        # Ensure Hunyuan3D models are unloaded before SDXL generation
        unload_hunyuan()

        checkpoint_config = CHECKPOINTS[checkpoint_name]

        # Load checkpoint if needed
        if loaded_checkpoint != checkpoint_name:
            worker_logger.info(f"Loading checkpoint: {checkpoint_config['filename']}")
            t0 = time.time()
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
                prediction_type=checkpoint_config["prediction_type"],
            )
            ckpt_path = hf_hub_download(settings.lightning_repo, checkpoint_config["filename"])
            state_dict = load_file(ckpt_path, device=device)
            pipe.unet.load_state_dict(state_dict)
            loaded_checkpoint = checkpoint_name
            worker_logger.info(f"Checkpoint loaded in {time.time() - t0:.2f}s")

        # Generate image
        worker_logger.info(f"Generating image for: {prompt[:50]}...")
        t0 = time.time()
        torch.cuda.synchronize() if "cuda" in device else None
        with torch.no_grad():
            results = pipe(prompt, num_inference_steps=checkpoint_config["steps"], guidance_scale=0)
        torch.cuda.synchronize() if "cuda" in device else None
        generated_image = results.images[0]
        worker_logger.info(f"Image generation: {time.time() - t0:.2f}s")

        # Preprocess image for 3D
        t0 = time.time()
        if do_remove_background:
            image = generated_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            processed_image = fill_background(image)
        else:
            processed_image = generated_image.convert("RGB")
        worker_logger.info(f"Preprocessing: {time.time() - t0:.2f}s")

        # Generate 3D based on selected engine
        worker_logger.info(f"Using 3D engine: {engine_3d}")

        # Prepare RGBA image for Hunyuan-based engines
        def prepare_image_for_hunyuan():
            if do_remove_background:
                img = remove_background(generated_image.convert("RGB"), rembg_session)
                return resize_foreground(img, foreground_ratio)
            else:
                return generated_image.convert("RGBA")

        if engine_3d == "hunyuan3d":
            # Hunyuan needs RGBA with transparent background
            image_for_3d = prepare_image_for_hunyuan()
            obj_path, glb_path = process_3d_hunyuan(image_for_3d, mesh_quality=mesh_quality)
        elif engine_3d == "hunyuan3d_mv":
            # Multi-view mode: use single image as front view
            # (Full multi-view only available for 3d-only endpoint with uploaded images)
            image_for_3d = prepare_image_for_hunyuan()
            mv_images = {"front": image_for_3d}
            obj_path, glb_path = process_3d_hunyuan_mv(mv_images, mesh_quality=mesh_quality)
        elif engine_3d == "hunyuan_api":
            # Hunyuan Cloud API
            image_for_3d = prepare_image_for_hunyuan()
            obj_path, glb_path = process_3d_hunyuan_api(image_for_3d, mesh_quality=mesh_quality)
        elif engine_3d == "tripo_api":
            # Cloud-based Tripo API
            image_for_3d = prepare_image_for_hunyuan()
            obj_path, glb_path = process_3d_tripo_api(image_for_3d)
        elif engine_3d == "auto_mv":
            # Auto Multi-View: Generated image → Gemini MV → Hunyuan3D-2 MV
            image_for_3d = prepare_image_for_hunyuan()
            obj_path, glb_path, mv_images = process_3d_gemini_mv(image_for_3d, mesh_quality=mesh_quality)
        else:  # triposr
            obj_path, glb_path = process_3d_triposr(processed_image, mc_resolution)
            mv_images = None

        total_time = time.time() - total_start
        worker_logger.info(f"=== TOTAL: {total_time:.2f}s ===")

        result = {
            "success": True,
            "generated_image": image_to_base64(generated_image),
            "processed_image": image_to_base64(processed_image),
            "mesh_obj_path": obj_path,
            "mesh_glb_path": glb_path,
            "processing_time": total_time,
            "engine_3d": engine_3d,
        }

        # Add multiview images if available
        if mv_images:
            result["multiview_front"] = image_to_base64(mv_images.get("front")) if mv_images.get("front") else None
            result["multiview_left"] = image_to_base64(mv_images.get("left")) if mv_images.get("left") else None
            result["multiview_right"] = image_to_base64(mv_images.get("right")) if mv_images.get("right") else None
            result["multiview_back"] = image_to_base64(mv_images.get("back")) if mv_images.get("back") else None

        return result

    def process_image_only(task: dict) -> dict:
        """Process image-only generation."""
        nonlocal loaded_checkpoint

        from .translator import translate_prompt

        prompt = task["prompt"]
        prompt = translate_prompt(prompt)
        checkpoint_name = task["checkpoint"]
        image_engine = task.get("image_engine", settings.default_image_engine)

        total_start = time.time()

        # Ensure Hunyuan3D models are unloaded before SDXL generation
        unload_hunyuan()

        # Generate image based on selected engine
        worker_logger.info(f"Using image engine: {image_engine}")

        if image_engine == "dalle":
            # Use DALL-E 3 API
            from .external_apis import get_dalle_client
            client = get_dalle_client()
            if client is None:
                raise Exception("OpenAI API key not configured. Set T2I3D_OPENAI_API_KEY environment variable.")
            worker_logger.info(f"[DALL-E 3] Generating image for: {prompt[:50]}...")
            generated_image = client.generate_image(prompt)

        elif image_engine == "gemini":
            # Use Gemini API
            from .external_apis import get_gemini_client
            client = get_gemini_client()
            if client is None:
                raise Exception("Gemini API key not configured. Set T2I3D_GEMINI_API_KEY environment variable.")
            worker_logger.info(f"[Gemini] Generating image for: {prompt[:50]}...")
            generated_image = client.generate_image(prompt)

        else:  # sdxl (default)
            checkpoint_config = CHECKPOINTS[checkpoint_name]

            # Load checkpoint if needed
            if loaded_checkpoint != checkpoint_name:
                worker_logger.info(f"Loading checkpoint: {checkpoint_config['filename']}")
                pipe.scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config,
                    timestep_spacing="trailing",
                    prediction_type=checkpoint_config["prediction_type"],
                )
                ckpt_path = hf_hub_download(settings.lightning_repo, checkpoint_config["filename"])
                state_dict = load_file(ckpt_path, device=device)
                pipe.unet.load_state_dict(state_dict)
                loaded_checkpoint = checkpoint_name

            # Generate image
            worker_logger.info(f"[SDXL-Lightning] Generating image for: {prompt[:50]}...")
            torch.cuda.synchronize() if "cuda" in device else None
            with torch.no_grad():
                results = pipe(prompt, num_inference_steps=checkpoint_config["steps"], guidance_scale=0)
            torch.cuda.synchronize() if "cuda" in device else None
            generated_image = results.images[0]

        return {
            "success": True,
            "image": image_to_base64(generated_image),
            "processing_time": time.time() - total_start,
            "image_engine": image_engine,
        }

    def process_3d_tripo_api(input_image: Image.Image) -> tuple:
        """Generate 3D using Tripo API (cloud-based)."""
        from .external_apis import get_tripo_client

        client = get_tripo_client()
        if client is None:
            raise Exception("Tripo API key not configured. Set T2I3D_TRIPO_API_KEY environment variable.")

        return client.image_to_3d(input_image)


    def process_3d_only(task: dict) -> dict:
        """Process 3D-only generation from image."""
        import base64
        from tsr.utils import remove_background, resize_foreground

        image_b64 = task["image"]
        image_left_b64 = task.get("image_left")  # Multi-view: left image
        image_right_b64 = task.get("image_right")  # Multi-view: right image
        image_back_b64 = task.get("image_back")  # Multi-view: back image
        do_remove_background = task["remove_background"]
        foreground_ratio = task["foreground_ratio"]
        mc_resolution = task["mc_resolution"]
        engine_3d = task.get("engine_3d", settings.default_3d_engine)
        mesh_quality = task.get("mesh_quality", "balanced")

        total_start = time.time()

        # Decode main (front) image
        image_data = base64.b64decode(image_b64)
        input_image = Image.open(io.BytesIO(image_data))

        # Decode optional multi-view images
        input_image_left = None
        input_image_right = None
        input_image_back = None
        if image_left_b64:
            left_data = base64.b64decode(image_left_b64)
            input_image_left = Image.open(io.BytesIO(left_data))
        if image_right_b64:
            right_data = base64.b64decode(image_right_b64)
            input_image_right = Image.open(io.BytesIO(right_data))
        if image_back_b64:
            back_data = base64.b64decode(image_back_b64)
            input_image_back = Image.open(io.BytesIO(back_data))

        # Preprocess main image
        if do_remove_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            processed_image = fill_background(image)
        else:
            processed_image = input_image.convert("RGB")

        # Generate 3D based on selected engine
        worker_logger.info(f"Using 3D engine: {engine_3d}")

        if engine_3d == "hunyuan3d_mv":
            # Multi-view Hunyuan3D-2
            mv_images = {"front": input_image}
            if input_image_left:
                mv_images["left"] = input_image_left
            if input_image_right:
                mv_images["right"] = input_image_right
            if input_image_back:
                mv_images["back"] = input_image_back
            obj_path, glb_path = process_3d_hunyuan_mv(mv_images, mesh_quality=mesh_quality)

        elif engine_3d == "hunyuan_api":
            # Hunyuan Cloud API
            if do_remove_background:
                image_for_3d = remove_background(input_image.convert("RGB"), rembg_session)
                image_for_3d = resize_foreground(image_for_3d, foreground_ratio)
            else:
                image_for_3d = input_image.convert("RGBA")
            obj_path, glb_path = process_3d_hunyuan_api(image_for_3d, mesh_quality=mesh_quality)
            mv_images = None

        elif engine_3d == "hunyuan3d":
            if do_remove_background:
                image_for_3d = remove_background(input_image.convert("RGB"), rembg_session)
                image_for_3d = resize_foreground(image_for_3d, foreground_ratio)
            else:
                image_for_3d = input_image.convert("RGBA")
            obj_path, glb_path = process_3d_hunyuan(image_for_3d, mesh_quality=mesh_quality)
            mv_images = None

        elif engine_3d == "tripo_api":
            # Use cloud-based Tripo API
            if do_remove_background:
                image_for_3d = remove_background(input_image.convert("RGB"), rembg_session)
                image_for_3d = resize_foreground(image_for_3d, foreground_ratio)
            else:
                image_for_3d = input_image.convert("RGBA")
            obj_path, glb_path = process_3d_tripo_api(image_for_3d)
            mv_images = None

        elif engine_3d == "gemini_mv" or engine_3d == "auto_mv":
            # Gemini multi-view → Hunyuan3D-2 MV pipeline
            # auto_mv in 3d-only context is the same as gemini_mv (image already exists)
            if do_remove_background:
                image_for_3d = remove_background(input_image.convert("RGB"), rembg_session)
                image_for_3d = resize_foreground(image_for_3d, foreground_ratio)
            else:
                image_for_3d = input_image.convert("RGBA")
            obj_path, glb_path, mv_images = process_3d_gemini_mv(image_for_3d, mesh_quality=mesh_quality)

        else:  # triposr
            obj_path, glb_path = process_3d_triposr(processed_image, mc_resolution)
            mv_images = None

        result = {
            "success": True,
            "processed_image": image_to_base64(processed_image),
            "mesh_obj_path": obj_path,
            "mesh_glb_path": glb_path,
            "processing_time": time.time() - total_start,
            "engine_3d": engine_3d,
        }

        # Add multiview images if available
        if mv_images:
            result["multiview_front"] = image_to_base64(mv_images.get("front")) if mv_images.get("front") else None
            result["multiview_left"] = image_to_base64(mv_images.get("left")) if mv_images.get("left") else None
            result["multiview_right"] = image_to_base64(mv_images.get("right")) if mv_images.get("right") else None
            result["multiview_back"] = image_to_base64(mv_images.get("back")) if mv_images.get("back") else None

        return result

    def process_part_segmentation(task: dict) -> dict:
        """Process part segmentation using P3-SAM."""
        mesh_path = task["mesh_path"]
        post_process = task.get("post_process", True)
        seed = task.get("seed", 42)

        total_start = time.time()

        # Unload other models to free GPU memory
        unload_hunyuan()

        try:
            from .part_segmentation import segment_mesh_parts, is_p3sam_available

            if not is_p3sam_available():
                return {
                    "success": False,
                    "error": "P3-SAM is not available. Please ensure hunyuan3d_part submodule is cloned.",
                }

            worker_logger.info(f"[P3-SAM] Starting part segmentation for: {mesh_path}")

            # Run segmentation
            segmented_path, aabb, face_ids = segment_mesh_parts(
                mesh_path=mesh_path,
                post_process=post_process,
                seed=seed,
            )

            part_count = len([i for i in np.unique(face_ids) if i >= 0])
            worker_logger.info(f"[P3-SAM] Found {part_count} parts")

            return {
                "success": True,
                "segmented_mesh_path": segmented_path,
                "part_count": part_count,
                "processing_time": time.time() - total_start,
            }

        except Exception as e:
            worker_logger.error(f"[P3-SAM] Part segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
            }

    # Main worker loop
    while True:
        try:
            task = task_q.get()

            if task is None:
                worker_logger.info("Received shutdown signal")
                break

            task_type = task.get("type")
            worker_logger.info(f"Processing task: {task_type}")

            try:
                if task_type == "text_to_3d":
                    result = process_text_to_3d(task)
                elif task_type == "image_only":
                    result = process_image_only(task)
                elif task_type == "3d_only":
                    result = process_3d_only(task)
                elif task_type == "health_check":
                    result = {
                        "success": True,
                        "gpu_available": torch.cuda.is_available(),
                        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "available_engines": ["triposr", "hunyuan3d", "hunyuan3d_mv", "hunyuan_api", "tripo_api", "gemini_mv", "auto_mv"],
                        "default_engine": settings.default_3d_engine,
                    }
                elif task_type == "segment_parts":
                    result = process_part_segmentation(task)
                else:
                    result = {"success": False, "error": f"Unknown task type: {task_type}"}

                result_q.put(result)

            except Exception as e:
                import traceback
                worker_logger.error(f"Task failed: {e}")
                traceback.print_exc()
                result_q.put({"success": False, "error": str(e)})

        except Exception as e:
            worker_logger.error(f"Worker loop error: {e}")


def start_worker() -> bool:
    """Start the CUDA worker process."""
    global task_queue, result_queue, worker_process

    if worker_process is not None and worker_process.is_alive():
        logger.warning("Worker already running")
        return True

    logger.info("Starting CUDA worker process...")

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    worker_process = mp.Process(
        target=cuda_worker_process,
        args=(task_queue, result_queue),
        daemon=True,
    )
    worker_process.start()

    # Wait for worker to be ready
    try:
        status = result_queue.get(timeout=settings.worker_timeout)
        if status.get("status") == "READY":
            logger.info("Worker is ready!")
            return True
        else:
            logger.error(f"Worker failed to start: {status.get('error')}")
            return False
    except Exception as e:
        logger.error(f"Worker startup timeout: {e}")
        return False


def stop_worker():
    """Stop the CUDA worker process."""
    global task_queue, result_queue, worker_process

    if task_queue is not None:
        task_queue.put(None)

    if worker_process is not None:
        worker_process.join(timeout=5)
        if worker_process.is_alive():
            worker_process.terminate()
        worker_process = None

    task_queue = None
    result_queue = None
    logger.info("Worker stopped")


def is_worker_running() -> bool:
    """Check if worker is running."""
    return worker_process is not None and worker_process.is_alive()
