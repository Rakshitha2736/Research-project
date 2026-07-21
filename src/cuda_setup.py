"""
CUDA helpers for LSTM training on NVIDIA GPUs (RTX 2050 / CUDA 12.6).

Use the project venv only:
  D:\\project\\Research Project\\.venv\\Scripts\\python.exe
"""

from __future__ import annotations

import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

# Tuned for RTX 2050 4GB + in-memory TensorDataset (270k x 30 x 8)
DEFAULT_BATCH_SIZE = 256
# TensorDataset is already in RAM — workers add Windows spawn overhead with little benefit
DEFAULT_NUM_WORKERS = 0


def require_cuda() -> torch.device:
    """Fail loudly if CUDA is missing (prevents silent CPU fallback)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this interpreter.\n"
            f"  sys.executable = {sys.executable}\n"
            f"  torch.__version__ = {torch.__version__}\n"
            "Use the project venv with CUDA PyTorch:\n"
            '  "D:\\project\\Research Project\\.venv\\Scripts\\python.exe"\n'
            "Do NOT use D:\\Programs\\Python\\python.exe (CPU-only build)."
        )
    torch.backends.cudnn.benchmark = True  # fixed seq_len=30 → find fastest conv/LSTM algos
    return torch.device("cuda")


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_gpu_diagnostics(model: torch.nn.Module | None = None, sample_batch=None) -> None:
    props = torch.cuda.get_device_properties(0)
    print("=== GPU DIAGNOSTICS ===")
    print(f"CUDA Available:     {torch.cuda.is_available()}")
    print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory (total): {props.total_memory / (1024**3):.2f} GB")
    print(f"Current Device:     {torch.cuda.current_device()}")
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"CUDA Version:       {torch.version.cuda}")
    print(f"cuDNN Enabled:      {torch.backends.cudnn.enabled}")
    print(f"cuDNN Benchmark:    {torch.backends.cudnn.benchmark}")
    print(f"Python Executable:  {sys.executable}")
    if model is not None:
        p = next(model.parameters())
        print(f"Model Device:       {p.device}")
    if sample_batch is not None:
        print(f"Batch Device:       {sample_batch.device}")
    print("========================")


def print_gpu_memory(tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    prefix = f"[{tag}] " if tag else ""
    print(
        f"{prefix}GPU Memory — allocated: {alloc:.1f} MB | "
        f"reserved: {reserved:.1f} MB | peak: {peak:.1f} MB"
    )


def make_loader(
    X,
    y,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    """
    DataLoader tuned for RTX 2050 + TensorDataset:
    - pin_memory=True: faster H2D copies with non_blocking=True
    - num_workers=0: data already in RAM; avoids Windows multiprocessing overhead
    """
    ds = TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
    kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def to_device(xb: torch.Tensor, yb: torch.Tensor, device: torch.device):
    return xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
