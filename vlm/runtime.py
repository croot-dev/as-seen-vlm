"""실행 환경(디바이스·dtype) 판별과 ZeroGPU 데코레이터 shim."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch

IS_ZERO_GPU = os.environ.get("SPACES_ZERO_GPU") is not None

def _noop_gpu(*args: Any, **kwargs: Any) -> Callable:
    """ZeroGPU 밖(로컬·CLI·일반 GPU Space)에서는 데코레이터가 아무것도 하지 않는다."""
    if args and callable(args[0]):
        return args[0]

    def decorator(fn: Callable) -> Callable:
        return fn

    return decorator


# ZeroGPU에서만 spaces를 import 한다. 여기서 실패하면 GPU를 못 받는 상태로
# 조용히 돌아가는 것보다 기동 시점에 터지는 편이 낫다.
if IS_ZERO_GPU:
    import spaces

    GPU = spaces.GPU
else:
    GPU = _noop_gpu


def _resolve_device() -> str:
    # ZeroGPU는 @GPU 함수에 진입하기 전까지 CUDA를 숨기므로 is_available()을 믿을 수 없다.
    if IS_ZERO_GPU or torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if IS_ZERO_GPU or torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


DEVICE = _resolve_device()
DTYPE = _resolve_dtype(DEVICE)
