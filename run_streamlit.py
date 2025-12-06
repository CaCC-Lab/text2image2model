#!/usr/bin/env python3
"""
Launcher script for Streamlit frontend.
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit frontend."""
    # Get the frontend module path
    frontend_path = Path(__file__).parent / "frontend_streamlit"
    app_path = frontend_path / "app.py"

    if not app_path.exists():
        print(f"Error: {app_path} not found")
        sys.exit(1)

    print("""
    ==========================================================
    |       Starting Streamlit Frontend                      |
    ==========================================================
    |  Make sure the backend is running: python run_backend.py
    ==========================================================
    """)

    # Build command with any additional arguments
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
    ] + sys.argv[1:]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down Streamlit frontend...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
