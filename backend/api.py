"""
FastAPI endpoints for Text-to-Image-to-3D pipeline.
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings, CHECKPOINTS
from .models import (
    GenerationRequest,
    GenerationResponse,
    ImageOnlyRequest,
    ImageOnlyResponse,
    ThreeDOnlyRequest,
    ThreeDOnlyResponse,
    HealthResponse,
    ConfigResponse,
    ErrorResponse,
    PartSegmentationRequest,
    PartSegmentationResponse,
)
from . import worker

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store mesh file paths for serving
mesh_files: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting application...")

    # Create directories
    os.makedirs(settings.output_dir, exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)
    os.makedirs(settings.static_dir, exist_ok=True)

    # Start worker
    if not worker.start_worker():
        logger.error("Failed to start worker!")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    worker.stop_worker()


# Create FastAPI app
app = FastAPI(
    title="Text-to-Image-to-3D API",
    description="API for generating images and 3D models from text prompts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error", detail=str(exc)).model_dump(),
    )


# Health check
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and worker health."""
    worker_running = worker.is_worker_running()

    gpu_info = {"gpu_available": False, "gpu_name": None}

    if worker_running and worker.task_queue is not None and worker.result_queue is not None:
        try:
            worker.task_queue.put({"type": "health_check"})
            result = worker.result_queue.get(timeout=10)
            if result.get("success"):
                gpu_info = {
                    "gpu_available": result.get("gpu_available", False),
                    "gpu_name": result.get("gpu_name"),
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")

    return HealthResponse(
        status="healthy" if worker_running else "unhealthy",
        worker_status="running" if worker_running else "stopped",
        **gpu_info,
    )


# Configuration
@app.get("/api/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """Get available configuration options."""
    return ConfigResponse(
        checkpoints=list(CHECKPOINTS.keys()),
        default_checkpoint=settings.default_checkpoint,
        default_remove_background=settings.default_remove_background,
        default_foreground_ratio=settings.default_foreground_ratio,
        default_mc_resolution=settings.default_mc_resolution,
        mc_resolution_range={"min": 32, "max": 512, "step": 32},
        foreground_ratio_range={"min": 0.5, "max": 1.0, "step": 0.05},
        available_3d_engines=["triposr", "hunyuan3d", "hunyuan3d_mv", "hunyuan_api", "tripo_api", "gemini_mv", "auto_mv"],
        available_image_engines=["sdxl", "dalle", "gemini"],
        default_image_engine=settings.default_image_engine,
        default_3d_engine=settings.default_3d_engine,
    )


# Full generation endpoint
@app.post("/api/generate", response_model=GenerationResponse, tags=["Generation"])
async def generate(request: GenerationRequest):
    """Generate image and 3D model from text prompt."""
    logger.info(f"Generate: worker_running={worker.is_worker_running()}, task_queue={worker.task_queue is not None}, result_queue={worker.result_queue is not None}")

    if not worker.is_worker_running():
        raise HTTPException(status_code=503, detail="Worker not running")

    if worker.task_queue is None or worker.result_queue is None:
        raise HTTPException(status_code=503, detail="Worker queues not initialized")

    logger.info(f"Generate request: {request.prompt[:50]}...")

    # Send task to worker
    worker.task_queue.put({
        "type": "text_to_3d",
        "prompt": request.prompt,
        "checkpoint": request.checkpoint,
        "remove_background": request.remove_background,
        "foreground_ratio": request.foreground_ratio,
        "mc_resolution": request.mc_resolution,
        "engine_3d": request.engine_3d,
    })

    # Wait for result (30 minutes for high quality generation)
    try:
        result = worker.result_queue.get(timeout=1800)
    except Exception as e:
        logger.error(f"Generation timeout: {e}")
        raise HTTPException(status_code=504, detail="Generation timeout")

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    # Store mesh paths for serving
    obj_filename = Path(result["mesh_obj_path"]).name
    glb_filename = Path(result["mesh_glb_path"]).name
    mesh_files[obj_filename] = result["mesh_obj_path"]
    mesh_files[glb_filename] = result["mesh_glb_path"]

    return GenerationResponse(
        success=True,
        generated_image=result["generated_image"],
        processed_image=result["processed_image"],
        mesh_obj_url=f"/api/models/{obj_filename}",
        mesh_glb_url=f"/api/models/{glb_filename}",
        processing_time=result["processing_time"],
        engine_3d=result.get("engine_3d"),
        multiview_front=result.get("multiview_front"),
        multiview_left=result.get("multiview_left"),
        multiview_right=result.get("multiview_right"),
        multiview_back=result.get("multiview_back"),
    )


# Image-only generation
@app.post("/api/generate/image-only", response_model=ImageOnlyResponse, tags=["Generation"])
async def generate_image_only(request: ImageOnlyRequest):
    """Generate image only from text prompt."""
    if not worker.is_worker_running():
        raise HTTPException(status_code=503, detail="Worker not running")

    if worker.task_queue is None or worker.result_queue is None:
        raise HTTPException(status_code=503, detail="Worker queues not initialized")

    logger.info(f"Image-only request: {request.prompt[:50]}...")

    worker.task_queue.put({
        "type": "image_only",
        "prompt": request.prompt,
        "checkpoint": request.checkpoint,
        "image_engine": request.image_engine,
    })

    try:
        result = worker.result_queue.get(timeout=120)
    except Exception as e:
        logger.error(f"Generation timeout: {e}")
        raise HTTPException(status_code=504, detail="Generation timeout")

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    return ImageOnlyResponse(
        success=True,
        image=result["image"],
        processing_time=result["processing_time"],
    )


# 3D-only generation
@app.post("/api/generate/3d-only", response_model=ThreeDOnlyResponse, tags=["Generation"])
async def generate_3d_only(request: ThreeDOnlyRequest):
    """Generate 3D model from uploaded image."""
    if not worker.is_worker_running():
        raise HTTPException(status_code=503, detail="Worker not running")

    if worker.task_queue is None or worker.result_queue is None:
        raise HTTPException(status_code=503, detail="Worker queues not initialized")

    logger.info("3D-only request received")

    worker.task_queue.put({
        "type": "3d_only",
        "image": request.image,
        "image_left": request.image_left,
        "image_right": request.image_right,
        "image_back": request.image_back,
        "remove_background": request.remove_background,
        "foreground_ratio": request.foreground_ratio,
        "mc_resolution": request.mc_resolution,
        "engine_3d": request.engine_3d,
        "mesh_quality": request.mesh_quality,
    })

    try:
        # 30 minute timeout for Hunyuan3D high quality texture generation
        result = worker.result_queue.get(timeout=1800)
    except Exception as e:
        logger.error(f"Generation timeout: {e}")
        raise HTTPException(status_code=504, detail="Generation timeout")

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    # Store mesh paths
    obj_filename = Path(result["mesh_obj_path"]).name
    glb_filename = Path(result["mesh_glb_path"]).name
    mesh_files[obj_filename] = result["mesh_obj_path"]
    mesh_files[glb_filename] = result["mesh_glb_path"]

    return ThreeDOnlyResponse(
        success=True,
        processed_image=result["processed_image"],
        mesh_obj_url=f"/api/models/{obj_filename}",
        mesh_glb_url=f"/api/models/{glb_filename}",
        processing_time=result["processing_time"],
        engine_3d=result.get("engine_3d"),
        multiview_front=result.get("multiview_front"),
        multiview_left=result.get("multiview_left"),
        multiview_right=result.get("multiview_right"),
        multiview_back=result.get("multiview_back"),
    )


# Part segmentation (post-processing)
@app.post("/api/segment-parts", response_model=PartSegmentationResponse, tags=["Generation"])
async def segment_parts(request: PartSegmentationRequest):
    """Segment a 3D mesh into parts using P3-SAM (post-processing step)."""
    if not worker.is_worker_running():
        raise HTTPException(status_code=503, detail="Worker not running")

    if worker.task_queue is None or worker.result_queue is None:
        raise HTTPException(status_code=503, detail="Worker queues not initialized")

    # Extract filename from URL
    mesh_url = request.mesh_glb_url
    filename = mesh_url.split("/")[-1]

    if filename not in mesh_files:
        raise HTTPException(status_code=404, detail="Mesh not found. Generate a 3D model first.")

    mesh_path = mesh_files[filename]
    if not os.path.exists(mesh_path):
        raise HTTPException(status_code=404, detail="Mesh file not found on disk")

    logger.info(f"Part segmentation request for: {filename}")

    worker.task_queue.put({
        "type": "segment_parts",
        "mesh_path": mesh_path,
        "post_process": request.post_process,
        "seed": request.seed,
    })

    try:
        # 10 minute timeout for part segmentation
        result = worker.result_queue.get(timeout=600)
    except Exception as e:
        logger.error(f"Part segmentation timeout: {e}")
        raise HTTPException(status_code=504, detail="Part segmentation timeout")

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    # Store segmented mesh path
    segmented_filename = Path(result["segmented_mesh_path"]).name
    mesh_files[segmented_filename] = result["segmented_mesh_path"]

    return PartSegmentationResponse(
        success=True,
        segmented_mesh_url=f"/api/models/{segmented_filename}",
        part_count=result.get("part_count"),
        processing_time=result["processing_time"],
    )


# Serve mesh files
@app.get("/api/models/{filename}", tags=["Files"])
async def get_model(filename: str):
    """Download generated 3D model file."""
    if filename not in mesh_files:
        raise HTTPException(status_code=404, detail="Model not found")

    file_path = mesh_files[filename]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "model/obj" if filename.endswith(".obj") else "model/gltf-binary"
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
    )


# Mount static files
@app.on_event("startup")
async def mount_static():
    """Mount static files directory."""
    static_path = Path(settings.static_dir)
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
