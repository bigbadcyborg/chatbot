"""Pydantic request/response models for the SoulForge API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CommandRequest(BaseModel):
    name: str
    args: str = ""


class CommandResponse(BaseModel):
    kind: str
    text: str = ""
    data: dict[str, Any] = {}
    success: bool = True


class PingResponse(BaseModel):
    status: str
    model: str
    ready: bool  # chat model loaded
    compute_backend: str
    stage: str = "ready"
    vision_loaded: bool = False
    loading: bool = False  # a session load (chat/agents/vision) is still running


class SessionStartRequest(BaseModel):
    chat_model: str | None = None  # None = keep configured default
    load_agents: bool = False
    load_vision: bool = False


class SessionStartResponse(BaseModel):
    started: bool
    message: str = ""


class AgentStartRequest(BaseModel):
    goal: str = ""
    run_id: str = ""  # used by resume


class AgentStartResponse(BaseModel):
    started: bool
    message: str = ""


class AgentStateResponse(BaseModel):
    running: bool = False
    stage: str = ""
    result: str = ""
    data: dict = {}


class AgentLoadStateResponse(BaseModel):
    loading: bool = False
    done: bool = False
    current: str = ""  # profile currently loading, e.g. "orchestrator"
    loaded_bytes: int = 0
    total_bytes: int = 0
    elapsed_seconds: float = 0.0
    eta_seconds: float = -1.0  # -1 = not yet estimable
    message: str = ""


class ShutdownResponse(BaseModel):
    stopping: bool
    message: str = ""


class SnapshotResponse(BaseModel):
    text: str
    success: bool = True


class TranscribeResponse(BaseModel):
    text: str
    success: bool = True
