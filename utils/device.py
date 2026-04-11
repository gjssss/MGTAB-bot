import torch


def resolve_device(preferred: str | None = "auto") -> torch.device:
    if preferred is None or preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preferred!r}")
