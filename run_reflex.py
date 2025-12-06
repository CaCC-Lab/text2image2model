#!/usr/bin/env python3
"""
Launcher script for Reflex frontend.
"""
import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run the Reflex frontend."""
    # Change to the frontend directory
    frontend_path = Path(__file__).parent / "frontend_reflex"

    if not frontend_path.exists():
        print(f"Error: {frontend_path} not found")
        sys.exit(1)

    print("""
    ==========================================================
    |        Starting Reflex Frontend                        |
    ==========================================================
    |  Make sure the backend is running: python run_backend.py
    ==========================================================
    """)

    # Build command
    cmd = [sys.executable, "-m", "reflex", "run"] + sys.argv[1:]

    try:
        # Run from the frontend directory
        subprocess.run(cmd, check=True, cwd=str(frontend_path))
    except KeyboardInterrupt:
        print("\nShutting down Reflex frontend...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Reflex frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
