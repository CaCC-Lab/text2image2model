#!/usr/bin/env python3
"""
Launcher script for Gradio frontend.
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Run the Gradio frontend."""
    # Get the frontend module path
    frontend_path = Path(__file__).parent / "frontend_gradio"
    app_path = frontend_path / "app.py"

    if not app_path.exists():
        print(f"Error: {app_path} not found")
        sys.exit(1)

    # Forward all arguments to the app
    cmd = [sys.executable, str(app_path)] + sys.argv[1:]

    print("""
    ==========================================================
    |       Starting Gradio Frontend                         |
    ==========================================================
    |  Make sure the backend is running: python run_backend.py
    ==========================================================
    """)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down Gradio frontend...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Gradio frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
