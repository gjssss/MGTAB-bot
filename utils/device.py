import torch


def _check_cuda_compat() -> None:
    """Ensure the current PyTorch build supports the installed GPU's compute capability."""
    cap = torch.cuda.get_device_capability()
    cap_tag = f"sm_{cap[0]}{cap[1]}"
    supported = torch.cuda.get_arch_list()
    if supported and cap_tag not in supported:
        name = torch.cuda.get_device_name()
        raise RuntimeError(
            f"GPU {name} ({cap_tag}) not supported by current PyTorch build "
            f"(supported arches: {supported}). Upgrade torch to a build that "
            f"includes kernels for {cap_tag}."
        )


def resolve_device(preferred: str | None = "auto") -> torch.device:
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            _check_cuda_compat()
            return torch.device("cuda")
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        _check_cuda_compat()
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preferred!r}")
