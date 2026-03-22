"""vm_micro.utils.paths
~~~~~~~~~~~~~~~~~~~~~~
Resolve data/output paths relative to the project root.
"""

from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    """Return the project root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT: Path = _project_root()


def resolve(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve *path* relative to *base* (defaults to project root).

    Absolute paths are returned as-is.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(base) if base is not None else PROJECT_ROOT
    return (root / p).resolve()


def ensure(path: str | Path, base: str | Path | None = None) -> Path:
    """Like :func:`resolve` but also creates the directory if it doesn't exist."""
    p = resolve(path, base)
    p.mkdir(parents=True, exist_ok=True)
    return p
