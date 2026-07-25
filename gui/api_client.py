"""HTTP/WebSocket client for the SoulForge API.

Kept free of Qt so it can be unit-tested against ``fastapi.testclient``. REST
calls go through an injectable ``httpx.Client`` (the tests pass a TestClient);
chat streaming uses the ``websockets`` sync client and yields decoded frames.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from gui.settings import GuiSettings


class ApiClient:
    """Thin wrapper over the SoulForge REST endpoints."""

    def __init__(self, settings: GuiSettings, http_client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._own_client = http_client is None
        # Generous timeout: commands like ingest or a curator review are blocking
        # LLM/IO work that can take minutes. (Agent runs use the async
        # /api/agents/* endpoints instead of one long request.)
        self._http = http_client or httpx.Client(base_url=settings.base_url, timeout=900.0)

    def _headers(self) -> dict[str, str]:
        if self.settings.auth_token:
            return {"X-SoulForge-Token": self.settings.auth_token}
        return {}

    def close(self) -> None:
        if self._own_client:
            self._http.close()

    def ping(self) -> dict[str, Any]:
        return self._http.get("/api/ping").json()

    def session_start(
        self, chat_model: str | None, load_agents: bool, load_vision: bool
    ) -> dict[str, Any]:
        resp = self._http.post(
            "/api/session/start",
            json={
                "chat_model": chat_model,
                "load_agents": load_agents,
                "load_vision": load_vision,
            },
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def commands(self) -> list[str]:
        resp = self._http.get("/api/commands", headers=self._headers())
        resp.raise_for_status()
        return resp.json()["commands"]

    def command(self, name: str, args: str = "") -> dict[str, Any]:
        resp = self._http.post(
            "/api/command",
            json={"name": name, "args": args},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def snapshot(self, image_png: bytes, prompt: str = "", inject: bool = True) -> dict[str, Any]:
        resp = self._http.post(
            "/api/snapshot",
            files={"image": ("snapshot.png", image_png, "image/png")},
            data={"prompt": prompt, "inject": str(inject).lower()},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def agents_start(self, goal: str) -> dict[str, Any]:
        resp = self._http.post(
            "/api/agents/start", json={"goal": goal}, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    def agents_resume(self, run_id: str = "") -> dict[str, Any]:
        resp = self._http.post(
            "/api/agents/resume", json={"run_id": run_id}, headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self) -> dict[str, Any]:
        """Ask the server to unload models and stop. Short timeout: it dies mid-call."""
        resp = self._http.post("/api/shutdown", headers=self._headers(), timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def agents_state(self) -> dict[str, Any]:
        resp = self._http.get("/api/agents/state", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def agents_load(self) -> dict[str, Any]:
        resp = self._http.post("/api/agents/load", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def agents_load_state(self) -> dict[str, Any]:
        resp = self._http.get("/api/agents/load-state", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def upload_doc(self, filename: str, content: bytes) -> dict[str, Any]:
        resp = self._http.post(
            "/api/rag/upload",
            files={"document": (filename, content, "application/octet-stream")},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    def transcribe(self, audio_wav: bytes, language: str = "") -> dict[str, Any]:
        resp = self._http.post(
            "/api/transcribe",
            files={"audio": ("clip.wav", audio_wav, "audio/wav")},
            data={"language": language},
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()


def stream_chat(ws_url: str, message: str) -> Iterator[dict[str, Any]]:
    """Connect to /ws/chat, send one message, and yield frames until 'done'.

    Imported lazily so the module stays importable without the ``websockets``
    dependency present (e.g. in REST-only test environments).
    """
    from websockets.sync.client import connect

    with connect(ws_url) as ws:
        ws.send(json.dumps({"message": message}))
        while True:
            raw = ws.recv()
            frame = json.loads(raw)
            yield frame
            if frame.get("type") == "done":
                break
