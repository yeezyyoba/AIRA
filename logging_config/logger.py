"""Structured logger configuration for AIRA.

Provides centralized logging across graph execution, LLM calls, tool executions,
and fallback events.
"""
import logging
import os
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root logger for AIRA."""
    logger = logging.getLogger("aira")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a namespaced logger under the 'aira' hierarchy."""
    if name:
        return logging.getLogger(f"aira.{name}")
    return logging.getLogger("aira")
