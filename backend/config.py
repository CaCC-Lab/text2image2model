"""
Configuration management for the backend.
"""
import os
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False

    # Device settings
    device: str = "cuda:0"
    use_gpu: bool = True

    # Model settings
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    lightning_repo: str = "ByteDance/SDXL-Lightning"
    triposr_model: str = "stabilityai/TripoSR"
    hunyuan3d_model: str = "tencent/Hunyuan3D-2"
    hunyuan3d_mv_model: str = "tencent/Hunyuan3D-2mv"  # Multi-view model

    # 3D Engine settings
    default_3d_engine: Literal["triposr", "hunyuan3d", "hunyuan3d_mv", "hunyuan_api", "tripo_api"] = "hunyuan3d"

    # Image generation settings
    default_image_engine: Literal["sdxl", "dalle", "gemini"] = "sdxl"

    # External API Keys (optional)
    tripo_api_key: str = ""  # Tripo3D API key for cloud 3D generation
    openai_api_key: str = ""  # OpenAI API key for DALL-E image generation
    gemini_api_key: str = ""  # Google Gemini API key for image generation
    hunyuan_secret_id: str = ""  # Tencent Cloud SecretID for Hunyuan API
    hunyuan_secret_key: str = ""  # Tencent Cloud SecretKey for Hunyuan API

    # Generation defaults
    default_checkpoint: Literal["1-Step", "2-Step", "4-Step", "8-Step"] = "4-Step"
    default_remove_background: bool = True
    default_foreground_ratio: float = 0.90  # Increased for better object visibility
    default_mc_resolution: int = 256  # Increased for better mesh quality
    default_texture_resolution: int = 1024  # UV-baked texture resolution for GLB export

    # Renderer settings
    chunk_size: int = 8192

    # File settings
    output_dir: str = "output"
    temp_dir: str = "temp"
    static_dir: str = "static"

    # CORS settings
    cors_origins: list[str] = ["*"]

    # Worker settings
    worker_timeout: int = 120

    class Config:
        env_prefix = "T2I3D_"
        env_file = ".env"


# Checkpoint configurations
CHECKPOINTS = {
    "1-Step": {
        "filename": "sdxl_lightning_1step_unet_x0.safetensors",
        "steps": 1,
        "prediction_type": "sample",
    },
    "2-Step": {
        "filename": "sdxl_lightning_2step_unet.safetensors",
        "steps": 2,
        "prediction_type": "epsilon",
    },
    "4-Step": {
        "filename": "sdxl_lightning_4step_unet.safetensors",
        "steps": 4,
        "prediction_type": "epsilon",
    },
    "8-Step": {
        "filename": "sdxl_lightning_8step_unet.safetensors",
        "steps": 8,
        "prediction_type": "epsilon",
    },
}

# Global settings instance
settings = Settings()
