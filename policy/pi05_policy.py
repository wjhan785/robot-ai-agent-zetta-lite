"""Thin client for the frozen Pi0.5 policy served by openpi (pi05_libero).

Runs ONLY on the TC1 workstation -- openpi + the checkpoint aren't
installed locally. Talks to an openpi policy server (started separately,
e.g. `python -m openpi.serve_policy ...` inside the same gpu_job.sh, or a
second job -- see PLAN.md §7/§8) over the openpi-client websocket protocol.

VERIFY on first use: the exact server entrypoint and the observation dict's
key names can vary across openpi versions/checkpoints. This wraps the
documented `openpi_client.websocket_client_policy.WebsocketClientPolicy`
API; fix up import paths/kwargs here once you can see real error messages
on TC1, not by guessing further offline.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict


class Pi05Policy:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        action_chunk: int = 10,
        exec_steps: int = 5,
    ):
        try:
            from openpi_client import websocket_client_policy  # type: ignore
        except ImportError as exc:  # pragma: no cover -- only fails off the GPU box
            raise ImportError(
                "openpi_client not installed -- Pi05Policy only runs on the TC1 "
                "workstation after slurm/setup_env.sh + the openpi install step."
            ) from exc

        self._client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.action_chunk = action_chunk
        self.exec_steps = exec_steps
        self._queue: Deque[Dict[str, Any]] = deque()

    def reset(self) -> None:
        self._queue.clear()

    def flush_chunk(self) -> None:
        """Discard any un-executed queued actions -- called after a
        recovery hands control back, so the VLA doesn't resume a stale
        chunk (PLAN.md §4 G4)."""
        self._queue.clear()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._queue:
            result = self._client.infer(obs)
            chunk = result["actions"][: self.exec_steps]
            self._queue.extend(chunk)
        return self._queue.popleft()
