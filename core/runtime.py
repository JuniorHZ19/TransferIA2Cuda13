"""Configuracion compartida de proveedores ONNX Runtime."""

import importlib
import os
import re


def get_onnx_providers(device="auto"):
    """Devuelve proveedores ONNX Runtime y el dispositivo seleccionado."""
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("El dispositivo debe ser 'auto', 'cuda' o 'cpu'")

    import onnxruntime as ort

    available = ort.get_available_providers()
    has_cuda = "CUDAExecutionProvider" in available
    if device == "cuda" and not has_cuda:
        raise RuntimeError(
            "Se solicito CUDA, pero ONNX Runtime no tiene CUDAExecutionProvider. "
            "Comprueba la instalacion de onnxruntime-gpu y las bibliotecas CUDA/cuDNN."
        )

    selected = "cuda" if device == "cuda" or (device == "auto" and has_cuda) else "cpu"
    if selected == "cpu":
        return ["CPUExecutionProvider"], selected

    _validate_cuda_runtime(ort)
    memory_limit_mb = int(os.environ.get("TRANSFERIA_GPU_MEM_LIMIT_MB", "2048"))
    if memory_limit_mb < 1:
        raise ValueError("TRANSFERIA_GPU_MEM_LIMIT_MB debe ser mayor que cero")

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "gpu_mem_limit": memory_limit_mb * 1024 * 1024,
            },
        ),
        "CPUExecutionProvider",
    ]
    return providers, selected


def _validate_cuda_runtime(ort):
    """Evita mezclar bibliotecas CUDA incompatibles de Torch y ONNX Runtime."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return

    torch_cuda = getattr(torch.version, "cuda", None)
    build_info = ort.get_build_info()
    match = re.search(r"CUDA_VERSION=(\d+)", build_info)
    ort_cuda_major = match.group(1) if match else None
    torch_cuda_major = str(torch_cuda).split(".", maxsplit=1)[0] if torch_cuda else None

    if ort_cuda_major and torch_cuda_major and ort_cuda_major != torch_cuda_major:
        raise RuntimeError(
            "Version CUDA incompatible entre ONNX Runtime "
            f"(CUDA {ort_cuda_major}) y PyTorch (CUDA {torch_cuda_major}). "
            "Reinstala ambos para la misma version CUDA; en Colab CUDA 13 usa "
            "PyTorch cu130 y onnxruntime-gpu 1.27 o superior."
        )