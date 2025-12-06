#!/usr/bin/env python3
"""
Launcher script for Next.js frontend.
"""
import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run the Next.js frontend."""
    # Get the frontend directory
    frontend_path = Path(__file__).parent / "frontend_nextjs"

    if not frontend_path.exists():
        print(f"Error: {frontend_path} not found")
        sys.exit(1)

    print("""
    ==========================================================
    |        Starting Next.js Frontend                       |
    ==========================================================
    |  Make sure the backend is running: python run_backend.py
    |  Frontend will be available at: http://localhost:3000
    ==========================================================
    """)

    # Check if node_modules exists
    node_modules = frontend_path / "node_modules"
    if not node_modules.exists():
        print("Installing dependencies...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=str(frontend_path),
                check=True,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            print("Please run 'npm install' manually in the frontend_nextjs directory")
            sys.exit(1)

    # Build command
    cmd = ["npm", "run", "dev"]
    if "--build" in sys.argv:
        cmd = ["npm", "run", "build"]
    elif "--start" in sys.argv:
        cmd = ["npm", "run", "start"]

    try:
        subprocess.run(cmd, cwd=str(frontend_path), check=True, shell=True)
    except KeyboardInterrupt:
        print("\nShutting down Next.js frontend...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Next.js frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
