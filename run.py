#!/usr/bin/env python3
"""
Unified launcher script for Text-to-Image-to-3D application.
Supports multiple frontend frameworks.
"""
import argparse
import subprocess
import sys
import os
import signal
import time
import requests
from pathlib import Path
from typing import Optional


FRONTENDS = {
    "gradio": {
        "name": "Gradio",
        "script": "run_gradio.py",
        "default_port": 7860,
        "description": "Gradio-based frontend with custom Sony theme",
    },
    "streamlit": {
        "name": "Streamlit",
        "script": "run_streamlit.py",
        "default_port": 8501,
        "description": "Streamlit-based frontend with modern UI",
    },
    "reflex": {
        "name": "Reflex",
        "script": "run_reflex.py",
        "default_port": 3000,
        "description": "Reflex-based frontend with reactive state management",
    },
    "nextjs": {
        "name": "Next.js",
        "script": "run_nextjs.py",
        "default_port": 3000,
        "description": "Next.js + React frontend with Three.js 3D viewer",
    },
}


def print_banner():
    """Print application banner."""
    print("""
    ==============================================================
    |                                                            |
    |          TEXT TO IMAGE TO 3D PIPELINE v1.0                 |
    |                                                            |
    |    Transform text into images and 3D models with AI        |
    |                                                            |
    ==============================================================
    """)


def print_help():
    """Print help information."""
    print("""
    Available frontends:
    """)
    for key, info in FRONTENDS.items():
        print(f"      {key:<12} - {info['description']}")
        print(f"                    Default port: {info['default_port']}")
        print()

    print("""
    Usage examples:
      python run.py                      # Start backend + Gradio (default)
      python run.py --frontend streamlit # Start backend + Streamlit
      python run.py --backend-only       # Start only the backend
      python run.py --frontend-only gradio  # Start only Gradio frontend
      python run.py --list               # List available frontends
    """)


def wait_for_backend(host: str, port: int, timeout: int = 120) -> bool:
    """Wait for backend to be ready."""
    url = f"http://{host}:{port}/health"
    start_time = time.time()

    print(f"Waiting for backend to be ready...")
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data.get("worker_status") == "running":
                    print(f"Backend is ready! (took {time.time() - start_time:.1f}s)")
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        # Show progress
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"  Still waiting... ({elapsed}s)")

    print(f"Backend failed to start within {timeout}s")
    return False


def run_backend(port: int = 8000, host: str = "127.0.0.1") -> subprocess.Popen:
    """Start the backend server."""
    script = Path(__file__).parent / "run_backend.py"
    cmd = [sys.executable, str(script), "--port", str(port), "--host", host]
    return subprocess.Popen(cmd)


def run_frontend(frontend: str, extra_args: list = None) -> subprocess.Popen:
    """Start a frontend server."""
    if frontend not in FRONTENDS:
        print(f"Error: Unknown frontend '{frontend}'")
        print(f"Available frontends: {', '.join(FRONTENDS.keys())}")
        sys.exit(1)

    script = Path(__file__).parent / FRONTENDS[frontend]["script"]
    if not script.exists():
        print(f"Error: Frontend script not found: {script}")
        sys.exit(1)

    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Unified launcher for Text-to-Image-to-3D application"
    )
    parser.add_argument(
        "--frontend", "-f",
        type=str,
        default="gradio",
        choices=list(FRONTENDS.keys()),
        help="Frontend to use (default: gradio)",
    )
    parser.add_argument(
        "--backend-only", "-b",
        action="store_true",
        help="Start only the backend server",
    )
    parser.add_argument(
        "--frontend-only",
        type=str,
        metavar="FRONTEND",
        help="Start only the specified frontend",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8080,
        help="Backend server port (default: 8080)",
    )
    parser.add_argument(
        "--backend-host",
        type=str,
        default="127.0.0.1",
        help="Backend server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available frontends",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode",
    )

    args = parser.parse_args()

    if args.list:
        print_help()
        return

    print_banner()

    processes = []

    try:
        # Start backend if not frontend-only
        if not args.frontend_only:
            print(f"Starting backend server on {args.backend_host}:{args.backend_port}...")
            backend_proc = run_backend(args.backend_port, args.backend_host)
            processes.append(backend_proc)

            if args.backend_only:
                print("Backend server started. Press Ctrl+C to stop.")
                backend_proc.wait()
                return

            # Wait for backend to be fully ready before starting frontend
            if not wait_for_backend(args.backend_host, args.backend_port):
                print("ERROR: Backend failed to start. Aborting.")
                for proc in processes:
                    proc.terminate()
                sys.exit(1)

        # Start frontend
        frontend = args.frontend_only or args.frontend
        print(f"Starting {FRONTENDS[frontend]['name']} frontend...")
        frontend_proc = run_frontend(frontend)
        processes.append(frontend_proc)

        # Wait for processes
        print("\nApplication started. Press Ctrl+C to stop all services.\n")
        for proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.wait()
        print("All services stopped.")

    except Exception as e:
        print(f"Error: {e}")
        for proc in processes:
            proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
