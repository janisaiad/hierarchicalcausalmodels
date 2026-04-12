"""Default compute devices: prefer GPU / MPS when available."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional


def default_torch_device_str() -> str:
    """Return ``cuda:0``, ``mps``, or ``cpu`` from env and hardware.

    Override with env ``HCM_DEFAULT_TORCH_DEVICE`` (e.g. ``cpu``, ``cuda:1``).
    """
    env = os.environ.get("HCM_DEFAULT_TORCH_DEVICE")
    if env is not None and str(env).strip():
        return str(env).strip()
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        mps_b = getattr(torch.backends, "mps", None)
        if mps_b is not None and mps_b.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_torch_device_from_mapping(
    torch_kwargs: Optional[Mapping[str, Any]],
) -> str:
    """Resolve ``device`` from ``torch_kwargs`` or use :func:`default_torch_device_str`."""
    if torch_kwargs is None:
        return default_torch_device_str()
    manual = torch_kwargs.get("device")
    if manual is None:
        return default_torch_device_str()
    return str(manual)
