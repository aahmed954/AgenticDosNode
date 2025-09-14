"""Main entry point for the Claude Code Proxy."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proxy_server import run_server

if __name__ == "__main__":
    run_server()