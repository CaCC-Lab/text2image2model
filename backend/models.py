"""
Pydantic models for API request/response validation.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request model for content generation."""

    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    checkpoint: Literal["1-Step", "2-Step", "4-Step", "8-Step"] = Field(
        default="4-Step", description="SDXL-Lightning checkpoint to use"
    )
    remove_background: bool = Field(default=True, description="Whether to remove background from generated image")
    foreground_ratio: float = Field(default=0.90, ge=0.5, le=1.0, description="Foreground ratio for preprocessing")
    mc_resolution: int = Field(default=256, ge=32, le=512, description="Marching cubes resolution for mesh extraction")
    engine_3d: Literal["triposr", "hunyuan3d", "tripo_api"] = Field(
        default="hunyuan3d", description="3D generation engine to use"
    )


class ImageOnlyRequest(BaseModel):
    """Request model for image-only generation."""

    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    checkpoint: Literal["1-Step", "2-Step", "4-Step", "8-Step"] = Field(
        default="4-Step", description="SDXL-Lightning checkpoint to use"
    )
    image_engine: Literal["sdxl", "dalle", "gemini"] = Field(
        default="sdxl", description="Image generation engine to use"
    )


class ThreeDOnlyRequest(BaseModel):
    """Request model for 3D-only generation from uploaded image."""

    image: str = Field(..., description="Base64 encoded image")
    remove_background: bool = Field(default=True, description="Whether to remove background")
    foreground_ratio: float = Field(default=0.90, ge=0.5, le=1.0, description="Foreground ratio for preprocessing")
    mc_resolution: int = Field(default=256, ge=32, le=512, description="Marching cubes resolution")
    engine_3d: Literal["triposr", "hunyuan3d", "tripo_api"] = Field(
        default="hunyuan3d", description="3D generation engine to use"
    )


class GenerationResponse(BaseModel):
    """Response model for content generation."""

    success: bool = Field(..., description="Whether generation was successful")
    generated_image: Optional[str] = Field(None, description="Base64 encoded generated image")
    processed_image: Optional[str] = Field(None, description="Base64 encoded processed image")
    mesh_obj_url: Optional[str] = Field(None, description="URL to download OBJ mesh")
    mesh_glb_url: Optional[str] = Field(None, description="URL to download GLB mesh")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    engine_3d: Optional[str] = Field(None, description="3D engine used for generation")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class ImageOnlyResponse(BaseModel):
    """Response model for image-only generation."""

    success: bool
    image: Optional[str] = Field(None, description="Base64 encoded image")
    processing_time: Optional[float] = None
    error: Optional[str] = None


class ThreeDOnlyResponse(BaseModel):
    """Response model for 3D-only generation."""

    success: bool
    processed_image: Optional[str] = Field(None, description="Base64 encoded processed image")
    mesh_obj_url: Optional[str] = None
    mesh_glb_url: Optional[str] = None
    processing_time: Optional[float] = None
    engine_3d: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: Literal["healthy", "unhealthy"]
    worker_status: Literal["running", "stopped", "unknown"]
    gpu_available: bool
    gpu_name: Optional[str] = None
    version: str = "1.0.0"


class ConfigResponse(BaseModel):
    """Response model for configuration info."""

    checkpoints: list[str]
    default_checkpoint: str
    default_remove_background: bool
    default_foreground_ratio: float
    default_mc_resolution: int
    mc_resolution_range: dict[str, int]
    foreground_ratio_range: dict[str, float]
    available_3d_engines: list[str]
    default_3d_engine: str
    available_image_engines: list[str]
    default_image_engine: str


class ErrorResponse(BaseModel):
    """Response model for errors."""

    success: bool = False
    error: str
    detail: Optional[str] = None
