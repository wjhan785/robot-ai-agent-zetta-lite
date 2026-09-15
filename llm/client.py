"""Thin DeepSeek API wrapper (PLAN.md: LLM = DeepSeek API).

DeepSeek's API is OpenAI-compatible (https://api.deepseek.com/chat/completions).
Only non-streaming text completion is used here -- diagnose/propose don't
need anything fancier. Swap providers by implementing LLMClient.complete().
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


@dataclass
class DeepSeekClient:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: Optional[str] = None
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set DEEPSEEK_API_KEY (e.g. sourced into the SLURM job's environment "
                "from a private file -- never commit it) before using DeepSeekClient."
            )

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        import requests  # local import: keeps this module importable without the dep on machines that never call it

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class FakeLLMClient:
    """Returns canned responses, cycled in order (repeating the last one
    once exhausted) -- for tests and the local dry run. Records every call
    so tests can assert on what was actually asked."""

    def __init__(self, responses: List[str]):
        if not responses:
            raise ValueError("FakeLLMClient needs at least one canned response")
        self.responses = list(responses)
        self.calls: List[Dict[str, str]] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user})
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]
