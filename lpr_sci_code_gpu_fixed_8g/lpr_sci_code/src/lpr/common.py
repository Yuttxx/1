from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
import yaml
from contextlib import nullcontext


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EasyConfig:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            value = self._data[item]
            if isinstance(value, dict):
                return EasyConfig(value)
            return value
        raise AttributeError(item)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        value = self._data[key]
        return EasyConfig(value) if isinstance(value, dict) else value


def get_project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(
    path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] | None = None,
    project_root_fallback: bool = True,
    must_exist: bool = False,
) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        candidate = raw
        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"Path not found: {candidate}")
        return candidate

    candidates: List[Path] = []

    def add_candidate(candidate: Path) -> None:
        if candidate not in candidates:
            candidates.append(candidate)

    if base_dir is not None:
        add_candidate(Path(base_dir).expanduser() / raw)
    add_candidate(Path.cwd() / raw)
    if project_root_fallback:
        add_candidate(PROJECT_ROOT / raw)

    if must_exist:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        attempted = "\n".join(f" - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"Path not found: {path}. Tried:\n{attempted}")

    return candidates[0]


def load_config(path: str | os.PathLike[str]) -> EasyConfig:
    cfg_path = resolve_path(path, must_exist=True)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return EasyConfig(cfg)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        requested = torch.device(prefer)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("[WARN] Requested CUDA but torch.cuda.is_available() is False. Falling back to CPU.")
            return torch.device("cpu")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_autocast(device: torch.device, enabled: bool = True):
    use_amp = bool(enabled and device.type == "cuda" and torch.cuda.is_available())
    if not use_amp:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=True)


def save_json(obj: Any, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(iterable: List[Any], batch_size: int) -> Iterator[List[Any]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def pad_sequences(seqs: List[List[int]], pad_value: int = 0) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def pad_float_sequences(seqs: List[List[float]], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.float32)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.float32)
    return out


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def softmax_masked(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    if mask.ndim == logits.ndim:
        invalid_rows = ~mask.any(dim=dim, keepdim=True)
        if invalid_rows.any():
            mask = mask.clone()
            mask = torch.where(invalid_rows, torch.ones_like(mask), mask)
    logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(logits, dim=dim)



def unwrap_state_dict(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise TypeError(f"Unsupported checkpoint type: {type(state)}")



def load_model_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str | os.PathLike[str],
    map_location: Any = "cpu",
    strict: bool = True,
    resize_mismatched: bool = False,
) -> Dict[str, Any]:
    state = unwrap_state_dict(torch.load(ckpt_path, map_location=map_location))
    if strict and not resize_mismatched:
        model.load_state_dict(state)
        return {"loaded": list(state.keys()), "resized": [], "skipped": []}

    current = model.state_dict()
    loaded: List[str] = []
    resized: List[str] = []
    skipped: List[str] = []
    merged = dict(current)
    for key, value in state.items():
        if key not in current:
            skipped.append(key)
            continue
        target = current[key]
        if value.shape == target.shape:
            merged[key] = value.to(dtype=target.dtype)
            loaded.append(key)
            continue
        if resize_mismatched and value.ndim == target.ndim:
            new_tensor = target.clone()
            slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, value.shape))
            new_tensor[slices] = value[slices].to(dtype=target.dtype)
            merged[key] = new_tensor
            resized.append(key)
            continue
        skipped.append(key)
    model.load_state_dict(merged, strict=False)
    return {"loaded": loaded, "resized": resized, "skipped": skipped}
