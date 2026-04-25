"""Structured logger — stderr output.

Provides get_logger() for consistent logging across the project.
Full JSON formatter will be implemented in Phase F.
"""

import logging
import sys


def get_logger(name: str = "smart-knowledge-hub") -> logging.Logger:
    """Return a logger that writes to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
