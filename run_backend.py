#!/usr/bin/env python
"""
Backend server launcher.
"""
import argparse
import multiprocessing as mp
import uvicorn

from backend.config import settings


def main():
    parser = argparse.ArgumentParser(description="Run the Text-to-Image-to-3D backend API")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (keep at 1 for GPU)")

    args = parser.parse_args()

    print(f"""
    ==========================================================
    |       Text-to-Image-to-3D Backend API Server           |
    ==========================================================
    |  Host: {args.host:<46} |
    |  Port: {args.port:<46} |
    |  Reload: {str(args.reload):<44} |
    ==========================================================
    """)

    uvicorn.run(
        "backend.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
