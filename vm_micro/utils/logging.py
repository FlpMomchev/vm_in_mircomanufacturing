"""vm_micro.utils.logging
~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around the standard library logger.
Call ``get_logger(__name__)`` at the top of every module.
"""

from __future__ import annotations

import logging
import sys


_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
