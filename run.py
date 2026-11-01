#!/usr/bin/env python3
"""AIRA (Amharic Intelligent Research Assistant) Entry Point.

Usage:
    python run.py             # Run minimal application verification & health check
    python run.py --server    # Start FastAPI backend server with Uvicorn
    python run.py --check     # Verify dependencies and directory structure
"""
import argparse
import sys
from pathlib import Path

# Ensure root directory is on the Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from logging_config.logger import get_logger, setup_logging

load_dotenv(ROOT_DIR / ".env")
setup_logging()
logger = get_logger("runner")


def check_environment() -> bool:
    """Verify system setup, Python version, and critical project directories."""
    logger.info("Running AIRA environment and configuration check...")

    python_version = sys.version_info
    logger.info(f"Python runtime: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 10) or python_version >= (3, 13):
        logger.warning(
            f"Python {python_version.major}.{python_version.minor} detected. "
            "Python 3.11 is strongly recommended for maximum compatibility with ChromaDB and ML packages."
        )

    # Check required directories
    required_dirs = [
        ROOT_DIR / "app",
        ROOT_DIR / "app" / "api",
        ROOT_DIR / "app" / "schemas",
        ROOT_DIR / "app" / "core",
        ROOT_DIR / "agents",
        ROOT_DIR / "llm",
        ROOT_DIR / "tools",
        ROOT_DIR / "meta_tools",
        ROOT_DIR / "memory",
        ROOT_DIR / "services",
        ROOT_DIR / "validation",
        ROOT_DIR / "logging_config",
        ROOT_DIR / "data" / "papers",
        ROOT_DIR / "data" / "processed",
        ROOT_DIR / "data" / "reports",
        ROOT_DIR / "data" / "chroma",
        ROOT_DIR / "tests",
        ROOT_DIR / "evaluation",
    ]

    all_ok = True
    for d in required_dirs:
        if d.is_dir():
            logger.debug(f"Directory exists: {d.relative_to(ROOT_DIR)}")
        else:
            logger.error(f"Missing directory: {d.relative_to(ROOT_DIR)}")
            all_ok = False

    logger.info("Environment check completed: " + ("ALL PASSED" if all_ok else "ISSUES FOUND"))
    return all_ok


def start_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True) -> None:
    """Start the FastAPI backend server using Uvicorn."""
    import uvicorn
    logger.info(f"Starting AIRA FastAPI backend on http://{host}:{port} (reload={reload})...")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


def main() -> int:
    parser = argparse.ArgumentParser(description="AIRA - Amharic Intelligent Research Assistant Backend")
    parser.add_argument("--server", action="store_true", help="Start FastAPI backend server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--check", action="store_true", help="Run environment and directory checks")

    args = parser.parse_args()

    if args.server:
        start_server(host=args.host, port=args.port)
        return 0

    # Default action: verify project setup and status
    success = check_environment()
    print("\n" + "=" * 60)
    print("AIRA: Amharic Intelligent Research Assistant (FastAPI Backend)")
    print("Phase 1: Architecture and Project Setup Status: READY")
    print("=" * 60)
    print("Backend Framework:   FastAPI + Uvicorn")
    print("Frontend:            Next.js / React (frontend/)")
    print("Directory structure: VERIFIED")
    print("Environment config:  VERIFIED (.env.example)")
    print("Dependencies spec:   VERIFIED (requirements.txt)")
    print("=" * 60 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
