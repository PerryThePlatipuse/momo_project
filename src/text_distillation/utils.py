from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=_json_default)
        file.write("\n")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def get_git_commit_hash(cwd: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def get_device(prefer_mps: bool = True) -> str:
    """Return the best available torch device string: cuda, mps, or cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and getattr(torch.backends, "mps", None) is not None:
        mps = torch.backends.mps
        is_built = getattr(mps, "is_built", lambda: True)
        if is_built() and mps.is_available():
            return "mps"
    return "cpu"


def move_batch_to_device(batch: dict[str, Any], device: str):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
