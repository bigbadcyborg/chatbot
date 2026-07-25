"""Tests for named model profile loading in ModelRuntime."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from app.core.chat_controller import ChatController
from app.core.config import load_config
from app.core.model_runtime import ModelRuntime


def _profile_config(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    default = models / "default.gguf"
    orchestrator = models / "orchestrator.gguf"
    creator = models / "creator.gguf"
    critic = models / "critic.gguf"
    for path in (default, orchestrator, creator, critic):
        path.write_bytes(path.name.encode())
    data = {
        "model": {
            "chatModelPath": str(default),
            "embeddingModelPath": str(models / "embed.gguf"),
            "gpuLayers": 0,
        },
        "features": {"agents": True, "rag": False, "memory": False},
        "agents": {
            "residencyMode": "hybrid",
            "modelProfiles": {
                "orchestrator": {
                    "chatModelPath": str(orchestrator),
                    "residency": "swap",
                    "temperature": 0.2,
                },
                "creator": {
                    "chatModelPath": str(creator),
                    "residency": "resident",
                    "maxTokens": 123,
                },
                "critic_executor": {
                    "chatModelPath": str(critic),
                    "residency": "resident",
                },
            },
        },
        "rag": {"dbPath": str(tmp_path / "chroma"), "docsPath": str(tmp_path / "docs")},
        "memory": {
            "userFile": str(tmp_path / "memory" / "user.md"),
            "memoryFile": str(tmp_path / "memory" / "memory.md"),
            "sessionFile": str(tmp_path / "memory" / "session.md"),
        },
        "skills": {
            "activePath": str(tmp_path / "skills" / "active"),
            "archivedPath": str(tmp_path / "skills" / "archived"),
            "registryPath": str(tmp_path / "skills" / "registry.json"),
        },
        "tasks": {"kanbanPath": str(tmp_path / "tasks" / "kanban.json")},
        "sessions": {"storePath": str(tmp_path / "sessions")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_named_profiles_resident_and_swap(monkeypatch, tmp_path) -> None:
    created = []

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(kwargs["model_path"])

        def create_chat_completion(self, messages, stream=False, **params):
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"{self.kwargs['model_path']}:{params.get('max_tokens')}"
                        }
                    }
                ]
            }

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    config = load_config(_profile_config(tmp_path))
    runtime = ModelRuntime(config)

    warnings = runtime.warm_resident_profiles()
    assert warnings == []
    assert len(runtime._chat_profiles) == 2

    response = runtime.create_chat_completion_for_profile(
        "creator",
        [{"role": "user", "content": "hi"}],
    )
    assert response["choices"][0]["message"]["content"].endswith(":123")
    assert len(runtime._chat_profiles) == 2

    runtime.create_chat_completion_for_profile(
        "orchestrator",
        [{"role": "user", "content": "plan"}],
    )
    assert list(runtime._chat_profiles) == ["orchestrator"]
    assert any("orchestrator.gguf" in path for path in created)

    # Loading a resident profile must evict the loaded swap profile so the
    # large orchestrator model does not stay co-resident with the workers.
    runtime.create_chat_completion_for_profile(
        "creator",
        [{"role": "user", "content": "build"}],
    )
    assert "orchestrator" not in runtime._chat_profiles
    assert "creator" in runtime._chat_profiles


class _PlanShim(ChatController):
    """ChatController without its heavy managers -- just config + runtime.

    agent_load_plan and the throughput helpers only touch those two.
    """

    def __init__(self, config) -> None:  # noqa: D401 - deliberately skips super
        self.config = config
        self.runtime = ModelRuntime(config)


def test_agent_load_plan_swap_planner_is_planner_only(tmp_path) -> None:
    """A swap planner evicts the workers, so the plan is just the planner."""
    config = load_config(_profile_config(tmp_path))
    ctrl = _PlanShim(config)

    plan = ctrl.agent_load_plan()

    assert [name for name, _p, _s in plan] == ["orchestrator"]
    assert all(size > 0 for _n, _p, size in plan)  # real file sizes


def test_load_throughput_roundtrip_with_ema(tmp_path) -> None:
    config = load_config(_profile_config(tmp_path))
    config.agents.runs_path = str(tmp_path / "runs")  # keep stats out of the repo
    ctrl = _PlanShim(config)

    assert ctrl.estimated_load_bps() == 0.0  # nothing remembered yet
    ctrl.record_load_bps(1000.0)
    assert ctrl.estimated_load_bps() == 1000.0
    ctrl.record_load_bps(2000.0)  # EMA: 0.5*1000 + 0.5*2000
    assert abs(ctrl.estimated_load_bps() - 1500.0) < 1e-6

