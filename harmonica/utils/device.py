"""Device detection and management for MPS/CUDA/CPU."""

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Get the best available device.

    Args:
        prefer: Device preference - "auto", "cuda", "mps", or "cpu"

    Returns:
        torch.device for computation
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    elif prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    elif prefer == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unknown device preference: {prefer}")


def device_info() -> dict:
    """Get information about available devices.

    Returns:
        Dictionary with device availability and details
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_available": True,
        "default_device": str(get_device()),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def supports_mixed_precision(device: torch.device) -> bool:
    """Check if device supports automatic mixed precision.

    Args:
        device: The device to check

    Returns:
        True if AMP is supported
    """
    if device.type == "cuda":
        return True
    elif device.type == "mps":
        # MPS has limited AMP support, enable with caution
        return True
    return False


def get_dtype_for_device(device: torch.device, use_mixed: bool = True) -> torch.dtype:
    """Get the appropriate dtype for a device.

    Args:
        device: Target device
        use_mixed: Whether to use mixed precision if available

    Returns:
        Appropriate torch.dtype
    """
    if not use_mixed:
        return torch.float32

    if device.type == "cuda":
        return torch.float16
    elif device.type == "mps":
        # MPS works better with float32 for stability
        return torch.float32
    return torch.float32
