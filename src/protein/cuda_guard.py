from __future__ import annotations

import subprocess
from typing import Any


def nvidia_smi_gpu_names() -> list[str]:
    """Return GPU names from nvidia-smi. Empty if nvidia-smi fails or no GPUs."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    lines = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
    return lines


def torch_cuda_probe() -> dict[str, Any]:
    import torch

    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_version_torch": getattr(torch.version, "cuda", None),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def ensure_cuda_for_openfold(*, purpose: str) -> dict[str, Any]:
    """
    Require an NVIDIA GPU visible to PyTorch CUDA — no CPU fallback.
    Raises RuntimeError if anything is missing.

    Used by **legacy** `scripts/protein/legacy/` only; ColabFold uses JAX in a separate env.
    """
    names = nvidia_smi_gpu_names()
    if not names:
        raise RuntimeError(
            f"{purpose}: nvidia-smi reported no GPUs (is the NVIDIA driver installed in WSL2 "
            "and is this session inside WSL?). Refusing CPU fallback."
        )

    probe = torch_cuda_probe()
    if not probe["cuda_available"]:
        raise RuntimeError(
            f"{purpose}: torch.cuda.is_available() is False "
            f"(nvidia-smi saw {names!r}). Install CUDA-enabled PyTorch in this environment, e.g. "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    if probe["cuda_device_count"] < 1:
        raise RuntimeError(f"{purpose}: torch.cuda.device_count() < 1; refusing to continue.")

    return {
        "purpose": purpose,
        "nvidia_smi_gpus": names,
        **probe,
    }


def ensure_nvidia_gpu_present(*, purpose: str) -> dict[str, Any]:
    """
    Require at least one GPU visible to nvidia-smi (no PyTorch / JAX import).
    Used for ColabFold / JAX inference running in a separate environment.
    """
    names = nvidia_smi_gpu_names()
    if not names:
        raise RuntimeError(
            f"{purpose}: nvidia-smi reported no GPUs. Refusing to continue "
            "(ColabFold needs a local GPU for structure inference in this smoke)."
        )
    return {"purpose": purpose, "nvidia_smi_gpus": names, "backend": "nvidia-smi-only"}
