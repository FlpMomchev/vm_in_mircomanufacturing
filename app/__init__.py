"""Local dashboard app layer.

This is intentionally small and simple:
- settings / paths
- SQLite state
- watcher
- runner, parser, Shiny UI
"""

from .db import DashboardDB
from .parser import RunParser
from .runner import FinalPredictionRunner, RunRequest, RunResult
from .settings import AppSettings, load_settings

__all__ = [
    "AppSettings",
    "DashboardDB",
    "FinalPredictionRunner",
    "RunRequest",
    "RunResult",
    "RunParser",
    "load_settings",
]
