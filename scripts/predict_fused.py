"""Backward-compatible entrypoint for vm-predict-fused.

This module keeps older console-script wrappers working after the
rename to ``scripts.final_prediction``.
"""

from __future__ import annotations

from scripts.final_prediction import main

if __name__ == "__main__":
    main()
