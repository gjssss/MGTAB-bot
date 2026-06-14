"""LLM configuration for optional bot-detection result analysis."""

import os
from dataclasses import dataclass


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(0, parsed)


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    api_base: str
    model: str
    api_key: str
    request_timeout: int
    max_retries: int

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.api_key.strip())

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            enabled=_parse_bool(os.getenv("ENABLE_LLM"), False),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.chatanywhere.org"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            request_timeout=_parse_int(os.getenv("LLM_REQUEST_TIMEOUT"), 30),
            max_retries=_parse_int(os.getenv("LLM_MAX_RETRIES"), 3),
        )

    def public_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "model": self.model,
        }
