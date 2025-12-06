"""
Backend module for Text-to-Image-to-3D pipeline.
Provides FastAPI endpoints and CUDA worker process.
"""

from .config import settings
from .worker import start_worker, stop_worker, task_queue, result_queue

__all__ = ["settings", "start_worker", "stop_worker", "task_queue", "result_queue"]
