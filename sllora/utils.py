"""Small shared helpers: seeding, device info, JSON I/O, timing."""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import subprocess
import tempfile

import numpy as np
import torch

GB = 1024 ** 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def gb(n_bytes: float) -> float:
    return round(float(n_bytes) / GB, 4)


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def save_json(obj, path: str) -> None:
    """Atomic write so a Colab disconnect never leaves a half-written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def free_cuda() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
