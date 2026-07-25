"""Chat controller: the non-UI heart of the chatbot.

Owns the model runtime, prompt assembly, memory, and RAG retrieval, and exposes
a small surface that both the CLI loop and the Textual TUI drive. Keeping this
logic UI-agnostic means the same code path powers every front end.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from app.agents.manager import AgentActionResult, AgentManager
from app.core.compute_backend import ComputeBackend
from app.core.config import (
    PROJECT_ROOT,
    AgentModelProfileConfig,
    AppConfig,
    resolve_path,
    save_agents,
    save_chat_model,
    save_onboarding,
    save_tools,
    save_vision,
)
from app.core.diagnostics import (
    format_config_view,
    format_diagnostics_view,
    format_health_view,
    run_startup_diagnostics,
)
from app.core.feature_state import FeatureStateManager
from app.core.model_catalog import (
    import_chat_model as catalog_import_chat_model,
    list_available_chat_models,
    path_for_config,
    resolve_chat_model_selection,
)
from app.core.model_runtime import ModelRuntime
from app.core.prompt_builder import PromptBuilder
from app.memory.episodic import (
    EpisodicMemoryResult,
    EpisodicMemoryStore,
    format_episodic_results,
)
from app.memory.memory_manager import MemoryManager, MemorySnapshot
from app.memory.memory_reviewer import (
    MemorySuggestion,
    format_suggestion_view,
    generate_suggestion,
    merge_and_compact,
)
from app.rag.ingest import IngestResult, ingest_documents
from app.rag.retriever import Retriever, RetrievedChunk, get_store_stats
from app.skills.curator import (
    ACTION_ARCHIVE,
    ACTION_COMPACT,
    CuratorFinding,
    CuratorReviewResult,
    format_review_view,
    generate_compaction,
    run_full_review,
)
from app.skills.skill_crystallizer import (
    SkillSuggestion,
    format_suggestion_view as format_skill_suggestion_view,
    generate_suggestion as generate_skill_suggestion,
    resolve_unique_name,
)
from app.skills.skill_manager import SkillManager
from app.skills.workflow_observer import WorkflowMarkResult, WorkflowObserver
from app.sessions.session_manager import SessionManager
from app.sessions.session_summarizer import generate_summary
from app.memory.memory_reviewer import _strip_user_turn
from app.tasks.task_manager import COLUMN_LABELS, TaskManager, normalize_column
from app.tasks.task_suggester import (
    ACTION_CREATE,
    ACTION_DELETE,
    ACTION_MOVE,
    ACTION_UPDATE,
    TaskSuggestion,
    format_suggestions_review,
    generate_suggestions,
)
from app.tools.executor import ToolExecutionContext, ToolExecutor
from app.tools.models import PendingToolCall, ToolCall, ToolResult, ToolTurnResult
from app.tools.parser import parse_tool_calls
from app.tools.permissions import is_tool_available, tool_risk
from app.tools.registry import (
    TOOL_EXAMPLE_ARGS,
    format_tools_catalog,
    list_tool_defs,
)
from app.tools.tool_log import log_tool_event, read_recent_log

SOUL_PATH = PROJECT_ROOT / "SOUL.md"


@dataclass
class TurnReviewResult:
    turn_count: int
    review_due: bool
    has_suggestion: bool
    message: str = ""


@dataclass
class SkillMarkResult:
    mark: WorkflowMarkResult
    has_suggestion: bool
    should_open_modal: bool
    message: str = ""


@dataclass
class SkillCrystallizeResult:
    has_suggestion: bool
    message: str = ""


@dataclass
class CuratorActionResult:
    success: bool
    message: str = ""


@dataclass
class TaskActionResult:
    success: bool
    message: str = ""


@dataclass
class TaskSuggestResult:
    suggestions: list[TaskSuggestion]
    message: str = ""

    @property
    def has_suggestions(self) -> bool:
        return bool(self.suggestions)


@dataclass
class ToolActionResult:
    success: bool
    message: str = ""
    result: ToolResult | None = None


@dataclass
class SessionActionResult:
    success: bool
    message: str = ""
    session_id: str = ""
    title: str = ""


@dataclass
class SessionSummaryResult:
    success: bool
    message: str = ""
    summary: str = ""
    truncated: bool = False


def load_soul() -> str:
    """Load the persona from SOUL.md, or fall back to a neutral default."""
    if not SOUL_PATH.exists():
        return "You are a helpful local chatbot."
    return SOUL_PATH.read_text(encoding="utf-8", errors="ignore").strip()


class ChatController:
    """Coordinates models, prompts, memory, and retrieval for a session."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runtime = ModelRuntime(config)
        self.prompt_builder = PromptBuilder(config)
        self.memory_manager = MemoryManager(config)
        self.episodic_memory = EpisodicMemoryStore(config, self.runtime)
        self.skill_manager = SkillManager(config)
        self.task_manager = TaskManager(config)
        self.session_manager = SessionManager(config)
        self.agent_manager = AgentManager(
            config,
            self.runtime,
            tool_executor_factory=self._tool_executor,
            retriever_provider=lambda: self.retriever,
            memory_manager=self.memory_manager,
            skill_manager=self.skill_manager,
        )
        self.workflow_observer = WorkflowObserver(config)
        self.features = FeatureStateManager(config, on_change=self._on_feature_change)

        self.turn_count: int = 0
        self.pending_suggestion: MemorySuggestion | None = None
        self.pending_skill_suggestion: SkillSuggestion | None = None
        self._pending_crystallize_fingerprint: str = ""
        self.pending_curator_findings: list[CuratorFinding] = []
        self._dismissed_curator_ids: set[str] = set()
        self.pending_task_suggestions: list[TaskSuggestion] = []
        self._dismissed_task_suggestion_ids: set[str] = set()
        self.pending_tool_calls: list[PendingToolCall] = []
        self.active_session_id: str | None = None

        # RAG is initialized lazily to avoid loading embedding model at startup
        self._retriever: Retriever | None = None
        self.selected_sources: list[str] | None = None  # None = all sources

        self.soul_text: str = ""
        self.memory: MemorySnapshot | None = None
        self.messages: list[dict[str, str]] = []
        self.last_retrieved_chunks: list[RetrievedChunk] = []
        self.last_episodic_results: list[EpisodicMemoryResult] = []
        self._last_user_input: str = ""
        self._last_assistant_reply: str = ""
        self.loaded: bool = False

    def _on_feature_change(self, key: str, enabled: bool) -> None:
        if key in ("soul", "memory", "skills", "kanban", "tools"):
            self._rebuild_system_prompt()
        elif key == "rag" and enabled and self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

    def _initialize_retriever(self) -> Retriever | None:
        """Lazily initialize the retriever (only when RAG is first enabled)."""
        if self._retriever is not None:
            return self._retriever
        try:
            self._retriever = Retriever(self.config, self.runtime)
            self.runtime.load_embedding_model()
            return self._retriever
        except Exception as error:  # noqa: BLE001
            print(f"[rag] Failed to initialize retriever: {error}")
            return None

    @property
    def retriever(self) -> Retriever | None:
        """Return retriever if RAG is enabled, initializing if needed."""
        if self.features.is_enabled("rag"):
            return self._initialize_retriever()
        return None

    @property
    def rag_enabled(self) -> bool:
        """Whether RAG retrieval is active (compat for existing call sites)."""
        return self.features.is_enabled("rag")

    def load(self) -> None:
        """Load models and assemble the initial system prompt (blocking)."""
        self.memory_manager.ensure_files()
        self.soul_text = load_soul() if self.features.is_enabled("soul") else ""
        self.memory = (
            self.memory_manager.load() if self.features.is_enabled("memory") else None
        )

        self.runtime.load_chat_model()

        if self.features.is_enabled("rag") and self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

        self.messages = [{"role": "system", "content": self._build_system_prompt()}]
        self.loaded = True

    def _build_system_prompt(self) -> str:
        skills: list[str] = []
        if self.features.is_enabled("skills"):
            # v1: Load all active skills for keyword matching or manual use.
            # In a later iteration, this will be more selective.
            active_skills = self.skill_manager.list_skills(status="active")
            for meta in active_skills:
                content = self.skill_manager.get_skill_content(meta["name"])
                if content:
                    skills.append(content)

        return self.prompt_builder.build_system_prompt(
            self.soul_text,
            self.memory,
            skills,
            task_summary=self._task_summary_for_prompt(),
        )

    def _task_summary_for_prompt(self) -> str:
        if not self.features.is_enabled("kanban"):
            return ""
        return self.task_manager.format_board_summary()

    def _rebuild_system_prompt(self) -> None:
        """Reload soul/memory content and refresh the system message."""
        self.soul_text = load_soul() if self.features.is_enabled("soul") else ""
        self.memory = (
            self.memory_manager.load() if self.features.is_enabled("memory") else None
        )
        if self.messages:
            self.messages[0] = {
                "role": "system",
                "content": self._build_system_prompt(),
            }

    def reload_soul(self) -> None:
        """Reload SOUL.md and rebuild the system prompt without restarting."""
        self._rebuild_system_prompt()

    def reload_memory(self) -> None:
        """Reload memory files and rebuild the system prompt without restarting."""
        self._rebuild_system_prompt()

    def list_chat_models(self) -> list[str]:
        """Return available chat model filenames from ``./models/``."""
        return [path.name for path in list_available_chat_models(self.config)]

    def format_model_list(self) -> str:
        """Return a human-readable model list with the active model marked."""
        current = self.config.model.chat_model.resolve()
        lines: list[str] = []
        for path in list_available_chat_models(self.config):
            marker = "*" if path.resolve() == current else " "
            lines.append(f"{marker} {path.name}")
        if not lines:
            return "No chat models found in ./models/. Use /model add <path> to import one."
        return "\n".join(lines)

    def _valid_agent_roles(self) -> str:
        return ", ".join(self.config.agents.roles)

    def _valid_agent_profiles(self) -> str:
        return ", ".join(self.config.agents.model_profiles)

    def _resolve_agent_role_name(self, role_name: str) -> str:
        key = role_name.strip().lower()
        if not key:
            raise ValueError(f"Role is required. Valid roles: {self._valid_agent_roles()}")
        roles = self.config.agents.roles
        if key in roles:
            return key
        matches = [name for name in roles if name.startswith(key)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(
                f"Ambiguous role '{role_name}'. Matches: {', '.join(matches)}"
            )
        raise ValueError(
            f"Unknown agent role '{role_name}'. Valid roles: {self._valid_agent_roles()}"
        )

    def _resolve_agent_profile_name(self, profile_name: str) -> str:
        query = profile_name.strip()
        if not query:
            raise ValueError(
                f"Profile is required. Valid profiles: {self._valid_agent_profiles()}"
            )
        profiles = self.config.agents.model_profiles
        if query in profiles:
            return query
        exact = [name for name in profiles if name.lower() == query.lower()]
        if len(exact) == 1:
            return exact[0]
        matches = [name for name in profiles if name.lower().startswith(query.lower())]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(
                f"Ambiguous profile '{profile_name}'. Matches: {', '.join(matches)}"
            )
        raise ValueError(
            f"Unknown agent profile '{profile_name}'. "
            f"Valid profiles: {self._valid_agent_profiles()}"
        )

    def _clone_agent_profile_template(
        self,
        profile: AgentModelProfileConfig | None,
    ) -> AgentModelProfileConfig:
        if profile is None:
            return AgentModelProfileConfig()
        return AgentModelProfileConfig(
            chat_model_path=profile.chat_model_path,
            residency=profile.residency,
            temperature=profile.temperature,
            top_p=profile.top_p,
            repeat_penalty=profile.repeat_penalty,
            max_tokens=profile.max_tokens,
            chat_format=profile.chat_format,
        )

    def _profile_used_by_other_roles(self, profile_name: str, role_name: str) -> bool:
        return any(
            name != role_name and role.model_profile == profile_name
            for name, role in self.config.agents.roles.items()
        )

    def _ensure_role_specific_profile(
        self,
        role_name: str,
    ) -> tuple[str, AgentModelProfileConfig]:
        role = self.config.agents.roles[role_name]
        profile_name = role.model_profile or self.config.agents.default_profile
        # A profile named after the role may already exist as a *shared*
        # profile (e.g. 'creator' serves both creator and synthesizer).
        # Mutating it would break the "does not change sibling roles"
        # guarantee, so shift to a dedicated '<role>_role' profile instead.
        target_name = role_name
        if self._profile_used_by_other_roles(target_name, role_name):
            target_name = f"{role_name}_role"
        target = self.config.agents.model_profiles.get(target_name)
        if target is None:
            template = self.config.agents.model_profiles.get(profile_name)
            if template is None:
                template = self.config.agents.model_profiles.get(
                    self.config.agents.default_profile
                )
            target = self._clone_agent_profile_template(template)
            self.config.agents.model_profiles[target_name] = target
        role.model_profile = target_name
        return target_name, target

    def _agent_profile_model_label(self, profile: AgentModelProfileConfig) -> str:
        if profile.chat_model is None:
            return f"inherits /model ({self.model_name})"
        return profile.chat_model.name

    def format_models_view(self) -> str:
        """Return chat and agent model routing in one view."""
        lines = [
            "Model routing:",
            f"  Chat model: {self.model_name}",
            f"  Chat path:  {self.config.model.chat_model}",
            "",
            "Available chat models:",
        ]
        available = list_available_chat_models(self.config)
        current = self.config.model.chat_model.resolve()
        if not available:
            lines.append("  (none found in ./models/; use /model add <path>)")
        for path in available:
            marker = "*" if path.resolve() == current else " "
            lines.append(f"  {marker} {path.name}")

        lines.extend(["", "Agent roles:"])
        for role_name, role in self.config.agents.roles.items():
            profile = self.config.agents.model_profiles.get(role.model_profile)
            if profile is None:
                lines.append(
                    f"  {role_name}: missing profile '{role.model_profile}'"
                )
                continue
            lines.append(
                f"  {role_name}: profile={role.model_profile}, "
                f"model={self._agent_profile_model_label(profile)}, "
                f"residency={profile.residency}"
            )

        lines.extend(["", "Agent profiles:"])
        for profile_name, profile in self.config.agents.model_profiles.items():
            lines.append(
                f"  {profile_name}: model={self._agent_profile_model_label(profile)}, "
                f"residency={profile.residency}"
            )

        vision = self.config.vision
        vision_label = vision.model.name if (vision.enabled and vision.model) else "(disabled)"
        lines.extend(["", "Vision model (snapshots):", f"  {vision_label}"])

        lines.extend(
            [
                "",
                "Commands:",
                "  /models chat <model>              Change the normal chat model",
                "  /models role <role> <model>       Set one agent role to a model",
                "  /models role <role> inherit       Make one role inherit /model",
                "  /models profile <profile> <model> Change a shared agent profile",
                "  /models vision <model> [mmproj]   Set the snapshot vision model",
                "  /models vision off                Disable the vision model",
            ]
        )
        return "\n".join(lines)

    def set_agent_role_model(
        self,
        role_name: str,
        model_selection: str,
        *,
        persist: bool = True,
    ) -> str:
        """Assign a chat model to one agent role without changing sibling roles."""
        role_key = self._resolve_agent_role_name(role_name)
        selection = model_selection.strip()
        if not selection:
            raise ValueError("Model name is required. Use /models role <role> <model>.")

        profile_name, profile = self._ensure_role_specific_profile(role_key)
        inherit_values = {"inherit", "default", "none", "null", "-"}
        if selection.lower() in inherit_values:
            profile.chat_model_path = None
            model_label = f"inherits /model ({self.model_name})"
        else:
            resolved = resolve_chat_model_selection(selection, self.config)
            if not resolved.exists():
                raise FileNotFoundError(f"Chat model not found: {resolved}")
            profile.chat_model_path = path_for_config(resolved)
            model_label = resolved.name

        self.runtime.unload_chat_profile(profile_name)
        if persist:
            save_agents(self.config)
        return (
            f"Role '{role_key}' now uses profile '{profile_name}' "
            f"with model: {model_label}. (saved to config.yaml)"
        )

    def set_agent_profile_model(
        self,
        profile_name: str,
        model_selection: str,
        *,
        persist: bool = True,
    ) -> str:
        """Assign a chat model to a shared agent model profile."""
        profile_key = self._resolve_agent_profile_name(profile_name)
        selection = model_selection.strip()
        if not selection:
            raise ValueError(
                "Model name is required. Use /models profile <profile> <model>."
            )

        profile = self.config.agents.model_profiles[profile_key]
        inherit_values = {"inherit", "default", "none", "null", "-"}
        if selection.lower() in inherit_values:
            profile.chat_model_path = None
            model_label = f"inherits /model ({self.model_name})"
        else:
            resolved = resolve_chat_model_selection(selection, self.config)
            if not resolved.exists():
                raise FileNotFoundError(f"Chat model not found: {resolved}")
            profile.chat_model_path = path_for_config(resolved)
            model_label = resolved.name

        self.runtime.unload_chat_profile(profile_key)
        if persist:
            save_agents(self.config)
        return (
            f"Profile '{profile_key}' now uses model: {model_label}. "
            "(saved to config.yaml)"
        )

    def _resolve_local_gguf(self, token: str) -> Path:
        """Resolve a GGUF path from an explicit path or a name under ./models."""
        token = token.strip().strip('"').strip("'")
        candidates = [
            resolve_path(token),
            resolve_path(f"./models/{token}"),
        ]
        for path in candidates:
            if path.exists():
                return path
        # Fall back to a unique partial match among ./models/*.gguf.
        models_dir = resolve_path("./models")
        if models_dir.exists():
            matches = [
                p for p in models_dir.glob("*.gguf") if token.lower() in p.name.lower()
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous model '{token}'. Matches: "
                    + ", ".join(p.name for p in matches)
                )
        raise FileNotFoundError(f"GGUF not found: {token}")

    def models_info(self) -> dict:
        """Structured model routing for the GUI (chat, roles, vision, available)."""
        available = [p.name for p in list_available_chat_models(self.config)]
        roles: dict[str, str] = {}
        for role_name, role in self.config.agents.roles.items():
            profile = self.config.agents.model_profiles.get(role.model_profile)
            roles[role_name] = (
                self._agent_profile_model_label(profile) if profile else "(missing)"
            )
        v = self.config.vision
        return {
            "chat_model": self.model_name,
            "available": available,
            "roles": roles,
            "vision": {
                "enabled": v.enabled,
                "model": v.model.name if (v.enabled and v.model) else "",
                "mmproj": v.mmproj.name if v.mmproj else "",
            },
        }

    # -- structured domain data for the GUI menus -----------------------

    def memory_sections(self) -> dict:
        """Editable raw content + char limits for each memory section."""
        limits = self.memory_manager.limits()
        sections = {}
        for name in ("user", "memory", "session"):
            sections[name] = {
                "content": self.memory_manager.read_raw(name),
                "limit": limits[name],
            }
        return {
            "injection_on": self.features.is_enabled("memory"),
            "sections": sections,
        }

    def sessions_data(self) -> dict:
        """Saved sessions with metadata for a picker."""
        return {
            "active_session_id": self.active_session_id,
            "sessions": [meta.to_dict() for meta in self.session_manager.list_sessions()],
        }

    def board_data(self) -> dict:
        """Kanban board as columns -> task dicts."""
        board = self.task_manager.list_board()
        return {
            "columns": [
                {
                    "key": column,
                    "label": COLUMN_LABELS.get(column, column),
                    "tasks": [task.to_dict() for task in tasks],
                }
                for column, tasks in board.items()
            ]
        }

    def agents_data(self, run_id: str = "") -> dict:
        """Current agent run (tasks/checkpoints) + list of all runs."""
        store = self.agent_manager.store
        if run_id:
            current = store.load(run_id)
        elif self.agent_manager.active_run_id:
            current = store.load(self.agent_manager.active_run_id) or store.latest()
        else:
            current = store.latest()
        runs = [
            {"run_id": r.run_id, "goal": r.goal, "status": r.status}
            for r in store.list_runs()
        ]
        return {
            "enabled": self.features.is_enabled("agents"),
            "current": current.to_dict() if current is not None else None,
            "runs": runs,
        }

    def curator_data(self) -> dict:
        """Pending curator findings + active/archived skills."""
        return {
            "enabled": self.features.is_enabled("curator"),
            "findings": [f.to_dict() for f in self.pending_curator_findings],
            "active_skills": self.skill_manager.list_skills(status="active"),
            "archived_skills": self.skill_manager.list_skills(status="archived"),
        }

    def save_uploaded_doc(self, filename: str, content: bytes) -> str:
        """Write an uploaded document into the RAG docs folder (basename only)."""
        safe = Path(filename).name  # strip any path components
        if not safe or safe in (".", ".."):
            raise ValueError("Invalid filename.")
        docs_dir = self.config.rag.docs_dir
        docs_dir.mkdir(parents=True, exist_ok=True)
        target = docs_dir / safe
        target.write_bytes(content)
        return f"Saved {safe} to docs/. Run ingest to index it."

    def delete_doc(self, name: str) -> str:
        """Delete a document from the RAG docs folder (basename only)."""
        safe = Path(name).name
        target = self.config.rag.docs_dir / safe
        if not target.exists():
            return f"Document not found: {safe}"
        target.unlink()
        return f"Deleted {safe} from docs/. Run ingest to update the index."

    def format_vision_view(self) -> str:
        """Human-readable current vision-model configuration."""
        v = self.config.vision
        if not v.enabled:
            return (
                "Vision model: (disabled)\n"
                "Set one with: /models vision <model.gguf> <mmproj.gguf>"
            )
        return "\n".join(
            [
                "Vision model:",
                f"  model:   {v.model_path}",
                f"  mmproj:  {v.mmproj_path or '(missing — required)'}",
                f"  handler: {v.chat_handler}",
                f"  evictChat: {v.evict_chat}",
            ]
        )

    def set_vision_model(
        self,
        model_selection: str,
        mmproj_selection: str | None = None,
        *,
        handler: str | None = None,
        persist: bool = True,
    ) -> str:
        """Point the vision model at a new GGUF (and optional mmproj), then persist."""
        model = self._resolve_local_gguf(model_selection)
        self.config.vision.model_path = path_for_config(model)
        if mmproj_selection:
            mmproj = self._resolve_local_gguf(mmproj_selection)
            self.config.vision.mmproj_path = path_for_config(mmproj)
        if handler:
            self.config.vision.chat_handler = handler
        self.runtime.unload_vision_model()
        if persist:
            save_vision(self.config)
        mmproj_note = (
            self.config.vision.mmproj_path or "(none set — add the mmproj file!)"
        )
        return (
            f"Vision model set to {model.name} (mmproj: {mmproj_note}, "
            f"handler: {self.config.vision.chat_handler}). Saved to config.yaml."
        )

    def set_vision_mmproj(self, mmproj_selection: str, *, persist: bool = True) -> str:
        """Set only the vision mmproj (CLIP projector) file."""
        mmproj = self._resolve_local_gguf(mmproj_selection)
        self.config.vision.mmproj_path = path_for_config(mmproj)
        self.runtime.unload_vision_model()
        if persist:
            save_vision(self.config)
        return f"Vision mmproj set to {mmproj.name}. Saved to config.yaml."

    def disable_vision(self, *, persist: bool = True) -> str:
        """Disable the vision model (snapshot falls back to reporting no model)."""
        self.config.vision.model_path = ""
        self.runtime.unload_vision_model()
        if persist:
            save_vision(self.config)
        return "Vision model disabled. Saved to config.yaml."

    def switch_chat_model(self, model_path: str | Path, *, persist: bool = True) -> str:
        """Unload the current chat model, load *model_path*, and optionally persist."""
        resolved = resolve_chat_model_selection(str(model_path), self.config)
        if not resolved.exists():
            raise FileNotFoundError(f"Chat model not found: {resolved}")

        current = self.config.model.chat_model.resolve()
        if resolved.resolve() == current:
            return resolved.name

        old_path = self.config.model.chat_model_path
        self.runtime.unload_chat_model()
        self.config.model.chat_model_path = path_for_config(resolved)

        try:
            self.runtime.load_chat_model()
        except Exception:
            self.config.model.chat_model_path = old_path
            try:
                self.runtime.load_chat_model()
            except Exception:
                self.loaded = False
            raise

        if persist:
            save_chat_model(self.config)
        self.loaded = True
        return self.model_name

    def import_chat_model(
        self,
        source_path: str | Path,
        *,
        switch_after: bool = False,
        persist: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> str:
        """Copy a ``.gguf`` into ``./models/`` and optionally switch to it."""
        source = Path(source_path).expanduser()
        dest = catalog_import_chat_model(source, on_progress=on_progress)
        message = f"Imported model as {dest.name}."
        if switch_after:
            name = self.switch_chat_model(dest, persist=persist)
            return f"{message} Switched to {name}."
        return message

    def get_memory_view(self) -> str:
        """Return formatted memory content with character counts."""
        self.memory_manager.ensure_files()
        snapshot = self.memory_manager.load()
        return self.memory_manager.format_view(
            snapshot,
            memory_enabled=self.features.is_enabled("memory"),
        )

    def save_memory(self, section: str, content: str) -> bool:
        """Save a memory section and reload the system prompt. Returns True if truncated."""
        _, truncated = self.memory_manager.save(section, content)
        self.reload_memory()
        return truncated

    def clear_all_memory(self) -> None:
        """Wipe user/memory/session files and episodic history, then rebuild."""
        for section in ("user", "memory", "session"):
            self.memory_manager.save(section, "")
        self.pending_suggestion = None
        self.episodic_memory.clear()
        self.last_episodic_results = []
        self.reload_memory()

    def clear_episodic_memory(self) -> str:
        """Forget stored conversation history (episodic memory) only."""
        count = self.episodic_memory.clear()
        self.last_episodic_results = []
        return f"Cleared {count} remembered conversation turn(s)."

    def search_episodic_memory(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[EpisodicMemoryResult]:
        """Search vector-backed conversation memory."""
        return self.episodic_memory.search(query, limit=limit)

    def recent_episodic_memory(self, *, limit: int = 10) -> list[EpisodicMemoryResult]:
        """Return recent vector-backed conversation memory."""
        return self.episodic_memory.recent(limit=limit)

    def format_memory_analysis(self, query: str = "") -> str:
        """Return static memory plus episodic memory results."""
        self.reload_memory()
        sections = [self.get_memory_view()]
        if query.strip():
            results = self.search_episodic_memory(query, limit=8)
            sections.append(
                f"--- episodic_memory search: {query.strip()} ---\n"
                f"{format_episodic_results(results)}"
            )
        else:
            results = self.recent_episodic_memory(limit=8)
            sections.append(
                "--- recent episodic_memory ---\n"
                f"{format_episodic_results(results)}"
            )
        return "\n\n".join(sections)

    def enable_memory(self) -> None:
        """Enable memory injection."""
        self.set_feature("memory", True)

    def disable_memory(self) -> None:
        """Disable memory injection."""
        self.set_feature("memory", False)

    def complete_turn(self) -> TurnReviewResult:
        """Increment turn counter and maybe generate a memory review suggestion."""
        if not self.features.is_enabled("memory"):
            return TurnReviewResult(self.turn_count, False, False)

        self.turn_count += 1
        self._store_episodic_turn()
        return self._run_memory_review_if_due()

    def _store_episodic_turn(self) -> None:
        if not (self._last_user_input.strip() or self._last_assistant_reply.strip()):
            return
        self.episodic_memory.add_turn(
            turn_id=str(uuid.uuid4()),
            user=self._last_user_input,
            assistant=self._last_assistant_reply,
            turn_count=self.turn_count,
        )

    def maybe_trigger_memory_review(self) -> bool:
        """Run review at the current turn count without incrementing."""
        result = self._run_memory_review_if_due()
        return result.has_suggestion and self.pending_suggestion is not None

    def _run_memory_review_if_due(self) -> TurnReviewResult:
        interval = self.config.memory.update_every_turns
        review_due = interval > 0 and self.turn_count % interval == 0

        if not review_due:
            return TurnReviewResult(self.turn_count, False, False)

        if self.pending_suggestion is not None:
            return TurnReviewResult(
                self.turn_count,
                True,
                True,
                "Memory review already pending. Run /memory-review to view.",
            )

        snapshot = self.memory_manager.load()
        suggestion, error = generate_suggestion(
            self.runtime,
            self.config,
            self.messages,
            snapshot,
            self.turn_count,
        )
        if error:
            return TurnReviewResult(
                self.turn_count,
                True,
                False,
                f"Memory review (turn {self.turn_count}) failed: {error}",
            )

        if suggestion is None:
            return TurnReviewResult(
                self.turn_count,
                True,
                False,
                f"Memory review (turn {self.turn_count}): no new facts to save.",
            )

        self.pending_suggestion = suggestion
        return TurnReviewResult(
            self.turn_count,
            True,
            True,
            f"Memory review ready (turn {self.turn_count}).",
        )

    def get_memory_review(self) -> str:
        """Return formatted pending suggestion, or a no-pending message."""
        if self.pending_suggestion is None:
            return "No pending memory suggestion."
        limit = self.memory_manager.limits()[self.pending_suggestion.section]
        return format_suggestion_view(self.pending_suggestion, limit)

    def accept_memory_suggestion(
        self,
        content: str | None = None,
    ) -> tuple[bool, bool]:
        """Save pending suggestion. Returns (saved, was_compacted)."""
        if self.pending_suggestion is None:
            raise ValueError("No pending memory suggestion.")

        section = self.pending_suggestion.section
        text = content if content is not None else self.pending_suggestion.proposed_content
        max_chars = self.memory_manager.limits()[section]
        final_text, was_compacted = merge_and_compact(
            self.runtime,
            section,
            "",
            text,
            max_chars,
        )
        _, truncated = self.memory_manager.save(section, final_text)
        self.reload_memory()
        self.pending_suggestion = None
        return True, was_compacted or truncated

    def reject_memory_suggestion(self) -> None:
        """Discard the pending memory suggestion without saving."""
        self.pending_suggestion = None

    def mark_workflow_success(self, note: str = "") -> SkillMarkResult:
        """Record a successful workflow from recent chat messages."""
        mark = self.workflow_observer.mark_success(
            self.messages,
            note=note,
            turn_count=self.turn_count,
        )
        if mark.already_crystallized:
            return SkillMarkResult(
                mark=mark,
                has_suggestion=False,
                should_open_modal=False,
                message=mark.message,
            )

        should_open_modal = False
        has_suggestion = self.pending_skill_suggestion is not None

        if mark.threshold_reached:
            self._pending_crystallize_fingerprint = mark.fingerprint
            if self.config.skills.auto_create:
                result = self.crystallize_workflow(mark.fingerprint)
                return SkillMarkResult(
                    mark=mark,
                    has_suggestion=result.has_suggestion,
                    should_open_modal=result.has_suggestion,
                    message=result.message or mark.message,
                )
            message = f"{mark.message} Run /crystallize to draft a skill."
            return SkillMarkResult(
                mark=mark,
                has_suggestion=False,
                should_open_modal=False,
                message=message,
            )

        return SkillMarkResult(
            mark=mark,
            has_suggestion=has_suggestion,
            should_open_modal=should_open_modal,
            message=mark.message,
        )

    def crystallize_workflow(
        self,
        fingerprint: str | None = None,
    ) -> SkillCrystallizeResult:
        """Generate a skill suggestion from a logged workflow."""
        target_fp = fingerprint or self._pending_crystallize_fingerprint
        workflow = self.workflow_observer.get_best_workflow_for_crystallize(target_fp)
        if workflow is None:
            return SkillCrystallizeResult(
                has_suggestion=False,
                message=(
                    "No workflow ready for crystallization. "
                    f"Mark success {self.config.skills.min_successful_repeats} times "
                    "with /success first."
                ),
            )

        if self.pending_skill_suggestion is not None:
            return SkillCrystallizeResult(
                has_suggestion=True,
                message="Skill suggestion already pending. Run /skill-accept or /skill-reject.",
            )

        existing_names = self.skill_manager.list_skill_names()
        suggestion, error = generate_skill_suggestion(
            self.runtime,
            self.config,
            self.messages,
            workflow,
            existing_names,
        )
        if error and suggestion is None:
            return SkillCrystallizeResult(
                has_suggestion=False,
                message=f"Skill crystallization failed: {error}",
            )

        if suggestion is None:
            return SkillCrystallizeResult(
                has_suggestion=False,
                message="Skill crystallization produced no suggestion.",
            )

        unique_name = resolve_unique_name(suggestion.name, set(existing_names))
        if unique_name != suggestion.name:
            suggestion.name = unique_name
            suggestion.proposed_content = re.sub(
                r"^name: .+$",
                f"name: {unique_name}",
                suggestion.proposed_content,
                count=1,
                flags=re.MULTILINE,
            )

        self.pending_skill_suggestion = suggestion
        self._pending_crystallize_fingerprint = workflow.fingerprint
        return SkillCrystallizeResult(
            has_suggestion=True,
            message=f"Skill suggestion ready for '{suggestion.name}'.",
        )

    def get_skill_review(self) -> str:
        if self.pending_skill_suggestion is None:
            return "No pending skill suggestion."
        return format_skill_suggestion_view(self.pending_skill_suggestion)

    def accept_skill_suggestion(self, content: str | None = None) -> bool:
        """Save pending skill suggestion. Requires skills feature enabled."""
        if self.pending_skill_suggestion is None:
            raise ValueError("No pending skill suggestion.")
        if not self.features.is_enabled("skills"):
            raise ValueError(
                "Skills feature is disabled. Run /features skills on before accepting."
            )

        suggestion = self.pending_skill_suggestion
        text = content if content is not None else suggestion.proposed_content
        saved = self.skill_manager.create_skill_from_suggestion(
            suggestion.name,
            suggestion.description,
            text,
            success_count=suggestion.success_count,
        )
        if not saved:
            raise ValueError(
                f"Could not save skill '{suggestion.name}' (it may already exist)."
            )

        if suggestion.fingerprint:
            self.workflow_observer.mark_crystallized(
                suggestion.fingerprint,
                suggestion.name,
            )

        self.pending_skill_suggestion = None
        self._pending_crystallize_fingerprint = ""
        self._rebuild_system_prompt()
        return True

    def reject_skill_suggestion(self) -> None:
        """Discard pending skill suggestion without saving."""
        self.pending_skill_suggestion = None

    def _visible_curator_findings(self) -> list[CuratorFinding]:
        self._prune_obsolete_curator_findings()
        return [
            f
            for f in self.pending_curator_findings
            if f.finding_id not in self._dismissed_curator_ids
            and self._finding_still_applicable(f)
        ]

    def run_curator_review(self) -> CuratorReviewResult:
        """Analyze active skills and store curator findings."""
        if not self.features.is_enabled("curator"):
            return CuratorReviewResult(
                findings=[],
                message="Curator is disabled. Run /features curator on.",
            )

        result = run_full_review(
            self.runtime,
            self.config,
            self.skill_manager,
            use_llm=True,
        )
        self.pending_curator_findings = result.findings
        self._dismissed_curator_ids.clear()
        return result

    def get_curator_review(self) -> str:
        visible = self._visible_curator_findings()
        return format_review_view(visible)

    def dismiss_curator_finding(self, finding_id: str) -> None:
        self._dismissed_curator_ids.add(finding_id)

    def _dismiss_findings_for_skill(self, skill_name: str) -> None:
        """Remove all pending findings targeting a skill (e.g. after archive)."""
        for finding in self.pending_curator_findings:
            if finding.skill_name == skill_name:
                self._dismissed_curator_ids.add(finding.finding_id)

    def _finding_still_applicable(self, finding: CuratorFinding) -> bool:
        """True when the skill is still active and the action can be applied."""
        if not self.skill_manager.is_active(finding.skill_name):
            return False
        if finding.proposed_action == ACTION_ARCHIVE:
            return True
        if finding.proposed_action == ACTION_COMPACT:
            return bool(finding.proposed_content) or bool(
                self.skill_manager.get_skill_content(finding.skill_name)
            )
        return True

    def _prune_obsolete_curator_findings(self) -> None:
        """Auto-dismiss findings that no longer apply (skill archived, etc.)."""
        for finding in self.pending_curator_findings:
            if finding.finding_id in self._dismissed_curator_ids:
                continue
            if not self._finding_still_applicable(finding):
                self._dismissed_curator_ids.add(finding.finding_id)

    def clear_curator_findings(self) -> None:
        self.pending_curator_findings = []
        self._dismissed_curator_ids.clear()

    def accept_curator_finding(self, finding_id: str) -> CuratorActionResult:
        """Apply a curator suggestion (archive or compact)."""
        if not self.features.is_enabled("curator"):
            return CuratorActionResult(
                success=False,
                message="Curator is disabled. Run /features curator on.",
            )

        self._prune_obsolete_curator_findings()

        finding = next(
            (f for f in self.pending_curator_findings if f.finding_id == finding_id),
            None,
        )
        if finding is None:
            return CuratorActionResult(
                success=False,
                message=f"Finding '{finding_id}' not found.",
            )

        if not self._finding_still_applicable(finding):
            self._dismiss_findings_for_skill(finding.skill_name)
            return CuratorActionResult(
                success=True,
                message=(
                    f"Finding for '{finding.skill_name}' no longer applies "
                    "(skill may already be archived). Cleared."
                ),
            )

        if finding.proposed_action == ACTION_ARCHIVE:
            if not self.skill_manager.archive_skill(finding.skill_name):
                return CuratorActionResult(
                    success=False,
                    message=f"Failed to archive skill '{finding.skill_name}'.",
                )
            self._dismiss_findings_for_skill(finding.skill_name)
            self._rebuild_system_prompt()
            return CuratorActionResult(
                success=True,
                message=f"Archived skill '{finding.skill_name}'.",
            )

        if finding.proposed_action == ACTION_COMPACT:
            content = finding.proposed_content
            if not content:
                content = self.skill_manager.get_skill_content(finding.skill_name) or ""
                content = generate_compaction(
                    self.runtime,
                    content,
                    self.config.curator.bloat_max_chars,
                )
            if not self.skill_manager.update_skill_content(finding.skill_name, content):
                if not self.skill_manager.is_active(finding.skill_name):
                    self._dismiss_findings_for_skill(finding.skill_name)
                    return CuratorActionResult(
                        success=True,
                        message=(
                            f"Skill '{finding.skill_name}' is archived; "
                            "compact finding cleared."
                        ),
                    )
                return CuratorActionResult(
                    success=False,
                    message=f"Failed to compact skill '{finding.skill_name}'.",
                )
            self._dismiss_findings_for_skill(finding.skill_name)
            self._rebuild_system_prompt()
            return CuratorActionResult(
                success=True,
                message=f"Compacted skill '{finding.skill_name}'.",
            )

        return CuratorActionResult(
            success=False,
            message=f"Unknown action '{finding.proposed_action}'.",
        )

    def archive_skill_direct(self, name: str) -> CuratorActionResult:
        """Fast-path archive without curator review."""
        if not name.strip():
            return CuratorActionResult(success=False, message="Skill name required.")
        skill_name = name.strip()
        if not self.skill_manager.archive_skill(skill_name):
            return CuratorActionResult(
                success=False,
                message=f"Failed to archive skill '{skill_name}'.",
            )
        self._dismiss_findings_for_skill(skill_name)
        self._rebuild_system_prompt()
        return CuratorActionResult(success=True, message=f"Archived skill '{skill_name}'.")

    def restore_skill_direct(self, name: str) -> CuratorActionResult:
        """Restore an archived skill to active."""
        if not name.strip():
            return CuratorActionResult(success=False, message="Skill name required.")
        skill_name = name.strip()
        if not self.skill_manager.restore_skill(skill_name):
            return CuratorActionResult(
                success=False,
                message=(
                    f"Failed to restore skill '{skill_name}'. "
                    "It may not exist or is already active."
                ),
            )
        self._rebuild_system_prompt()
        return CuratorActionResult(
            success=True,
            message=f"Restored skill '{skill_name}' to active.",
        )

    def compact_skill_direct(self, name: str) -> CuratorReviewResult:
        """Fast-path compaction: generate a single bloat finding for a skill."""
        if not self.features.is_enabled("curator"):
            return CuratorReviewResult(
                findings=[],
                message="Curator is disabled. Run /features curator on.",
            )

        skill_name = name.strip()
        if not skill_name:
            return CuratorReviewResult(
                findings=[],
                message="Skill name required. Usage: /curator-compact <skill>",
            )

        content = self.skill_manager.get_skill_content(skill_name)
        if not content:
            return CuratorReviewResult(
                findings=[],
                message=f"Skill '{skill_name}' not found.",
            )

        compacted = generate_compaction(
            self.runtime,
            content,
            self.config.curator.bloat_max_chars,
        )
        finding = CuratorFinding(
            finding_id=uuid.uuid4().hex[:12],
            finding_type="bloat",
            skill_name=skill_name,
            rationale=f"Compact '{skill_name}' from {len(content)} to target size.",
            proposed_action=ACTION_COMPACT,
            proposed_content=compacted,
        )
        self.pending_curator_findings = [finding]
        self._dismissed_curator_ids.clear()
        return CuratorReviewResult(
            findings=[finding],
            message=f"Compaction draft ready for '{skill_name}'.",
        )

    def get_board_view(self) -> str:
        if not self.features.is_enabled("kanban"):
            return "Kanban is disabled. Run /features kanban on."
        return self.task_manager.format_board_view()

    def create_task_direct(
        self,
        title: str,
        description: str = "",
    ) -> TaskActionResult:
        if not self.features.is_enabled("kanban"):
            return TaskActionResult(
                success=False,
                message="Kanban is disabled. Run /features kanban on.",
            )
        if not title.strip():
            return TaskActionResult(success=False, message="Task title required.")
        task = self.task_manager.create_task(title, description)
        if task is None:
            return TaskActionResult(success=False, message="Failed to create task.")
        self._rebuild_system_prompt()
        return TaskActionResult(
            success=True,
            message=f"Created task [{task.id}] '{task.title}' in Backlog.",
        )

    def move_task_direct(self, task_id: str, column: str) -> TaskActionResult:
        if not self.features.is_enabled("kanban"):
            return TaskActionResult(
                success=False,
                message="Kanban is disabled. Run /features kanban on.",
            )
        if not task_id.strip():
            return TaskActionResult(
                success=False,
                message="Usage: /task-move <id> <column>",
            )
        normalized = normalize_column(column)
        if normalized is None:
            return TaskActionResult(
                success=False,
                message=(
                    f"Unknown column '{column}'. "
                    "Use backlog, in_progress, blocked, or done."
                ),
            )
        located = self.task_manager.get_task(task_id.strip())
        if located is None:
            return TaskActionResult(
                success=False,
                message=f"Task '{task_id}' not found.",
            )
        _, task = located
        if not self.task_manager.move_task(task.id, normalized):
            return TaskActionResult(
                success=False,
                message=f"Failed to move task '{task.id}'.",
            )
        self._prune_obsolete_task_suggestions()
        self._rebuild_system_prompt()
        label = COLUMN_LABELS.get(normalized, normalized)
        return TaskActionResult(
            success=True,
            message=f"Moved task [{task.id}] '{task.title}' to {label}.",
        )

    def delete_task_direct(self, task_id: str) -> TaskActionResult:
        if not self.features.is_enabled("kanban"):
            return TaskActionResult(
                success=False,
                message="Kanban is disabled. Run /features kanban on.",
            )
        if not task_id.strip():
            return TaskActionResult(success=False, message="Task ID required.")
        located = self.task_manager.get_task(task_id.strip())
        if located is None:
            return TaskActionResult(
                success=False,
                message=f"Task '{task_id}' not found.",
            )
        _, task = located
        if not self.task_manager.delete_task(task.id):
            return TaskActionResult(
                success=False,
                message=f"Failed to delete task '{task.id}'.",
            )
        self._dismiss_suggestions_for_task(task.id)
        self._rebuild_system_prompt()
        return TaskActionResult(
            success=True,
            message=f"Deleted task [{task.id}] '{task.title}'.",
        )

    def _suggestion_still_applicable(self, suggestion: TaskSuggestion) -> bool:
        if suggestion.action == ACTION_CREATE:
            return bool(suggestion.title.strip())
        if suggestion.action in (ACTION_MOVE, ACTION_UPDATE, ACTION_DELETE):
            return self.task_manager.get_task(suggestion.task_id) is not None
        return False

    def _prune_obsolete_task_suggestions(self) -> None:
        for suggestion in self.pending_task_suggestions:
            if suggestion.suggestion_id in self._dismissed_task_suggestion_ids:
                continue
            if not self._suggestion_still_applicable(suggestion):
                self._dismissed_task_suggestion_ids.add(suggestion.suggestion_id)

    def _visible_task_suggestions(self) -> list[TaskSuggestion]:
        self._prune_obsolete_task_suggestions()
        return [
            suggestion
            for suggestion in self.pending_task_suggestions
            if suggestion.suggestion_id not in self._dismissed_task_suggestion_ids
            and self._suggestion_still_applicable(suggestion)
        ]

    def _dismiss_suggestions_for_task(self, task_id: str) -> None:
        for suggestion in self.pending_task_suggestions:
            if suggestion.task_id == task_id:
                self._dismissed_task_suggestion_ids.add(suggestion.suggestion_id)

    def get_task_suggestions_review(self) -> str:
        return format_suggestions_review(self._visible_task_suggestions())

    def dismiss_task_suggestion(self, suggestion_id: str) -> None:
        self._dismissed_task_suggestion_ids.add(suggestion_id)

    def clear_task_suggestions(self) -> None:
        self.pending_task_suggestions = []
        self._dismissed_task_suggestion_ids.clear()

    def run_task_suggest(self) -> TaskSuggestResult:
        if not self.features.is_enabled("kanban"):
            return TaskSuggestResult(
                suggestions=[],
                message="Kanban is disabled. Run /features kanban on.",
            )
        suggestions = generate_suggestions(
            self.runtime,
            self.task_manager,
            self.messages,
            self.config,
        )
        self.pending_task_suggestions = suggestions
        self._dismissed_task_suggestion_ids.clear()
        if not suggestions:
            return TaskSuggestResult(
                suggestions=[],
                message="No task updates suggested from recent conversation.",
            )
        return TaskSuggestResult(
            suggestions=suggestions,
            message=f"{len(suggestions)} task suggestion(s) ready for review.",
        )

    def accept_task_suggestion(self, suggestion_id: str) -> TaskActionResult:
        if not self.features.is_enabled("kanban"):
            return TaskActionResult(
                success=False,
                message="Kanban is disabled. Run /features kanban on.",
            )

        self._prune_obsolete_task_suggestions()
        suggestion = next(
            (
                item
                for item in self.pending_task_suggestions
                if item.suggestion_id == suggestion_id
            ),
            None,
        )
        if suggestion is None:
            return TaskActionResult(
                success=False,
                message=f"Suggestion '{suggestion_id}' not found.",
            )

        if not self._suggestion_still_applicable(suggestion):
            self._dismissed_task_suggestion_ids.add(suggestion_id)
            return TaskActionResult(
                success=True,
                message="Suggestion no longer applies. Cleared.",
            )

        if suggestion.action == ACTION_CREATE:
            task = self.task_manager.create_task(
                suggestion.title,
                suggestion.description,
                suggestion.target_column or "backlog",
            )
            if task is None:
                return TaskActionResult(success=False, message="Failed to create task.")
            self._dismissed_task_suggestion_ids.add(suggestion_id)
            self._rebuild_system_prompt()
            return TaskActionResult(
                success=True,
                message=f"Created task [{task.id}] '{task.title}'.",
            )

        if suggestion.action == ACTION_MOVE:
            if not self.task_manager.move_task(
                suggestion.task_id,
                suggestion.target_column,
            ):
                return TaskActionResult(success=False, message="Failed to move task.")
            self._dismiss_suggestions_for_task(suggestion.task_id)
            self._rebuild_system_prompt()
            return TaskActionResult(
                success=True,
                message=f"Moved task [{suggestion.task_id}] to {suggestion.target_column}.",
            )

        if suggestion.action == ACTION_UPDATE:
            kwargs: dict[str, str] = {}
            if suggestion.title:
                kwargs["title"] = suggestion.title
            if suggestion.description:
                kwargs["description"] = suggestion.description
            if not self.task_manager.update_task(suggestion.task_id, **kwargs):
                return TaskActionResult(success=False, message="Failed to update task.")
            self._dismiss_suggestions_for_task(suggestion.task_id)
            self._rebuild_system_prompt()
            return TaskActionResult(
                success=True,
                message=f"Updated task [{suggestion.task_id}].",
            )

        if suggestion.action == ACTION_DELETE:
            located = self.task_manager.get_task(suggestion.task_id)
            title = located[1].title if located else suggestion.task_id
            if not self.task_manager.delete_task(suggestion.task_id):
                return TaskActionResult(success=False, message="Failed to delete task.")
            self._dismiss_suggestions_for_task(suggestion.task_id)
            self._rebuild_system_prompt()
            return TaskActionResult(
                success=True,
                message=f"Deleted task [{suggestion.task_id}] '{title}'.",
            )

        return TaskActionResult(
            success=False,
            message=f"Unknown action '{suggestion.action}'.",
        )

    def get_conversation_messages(self) -> list[dict[str, str]]:
        """Return user/assistant messages from the active conversation."""
        return [
            message
            for message in self.messages
            if message.get("role") in ("user", "assistant")
        ]

    @staticmethod
    def message_for_display(message: dict[str, str]) -> str:
        """Strip wrappers from stored user turns for TUI display."""
        content = str(message.get("content", ""))
        if message.get("role") == "user":
            return _strip_user_turn(content)
        return content

    def _clear_pending_state(self) -> None:
        self.pending_suggestion = None
        self.pending_skill_suggestion = None
        self._pending_crystallize_fingerprint = ""
        self.pending_curator_findings = []
        self._dismissed_curator_ids.clear()
        self.pending_task_suggestions = []
        self._dismissed_task_suggestion_ids.clear()

    def list_sessions_view(self) -> str:
        return self.session_manager.format_session_list()

    def save_session_direct(self, title: str = "") -> SessionActionResult:
        conversation = self.get_conversation_messages()
        if not conversation:
            return SessionActionResult(
                success=False,
                message="Nothing to save. Chat first, then run /session-save.",
            )

        existing_summary = ""
        if self.active_session_id:
            existing = self.session_manager.get_session(self.active_session_id)
            if existing:
                existing_summary = existing.summary

        saved = self.session_manager.save_session(
            self.messages,
            title=title,
            summary=existing_summary,
            turn_count=self.turn_count,
            session_id=self.active_session_id,
        )
        if saved is None:
            return SessionActionResult(success=False, message="Failed to save session.")
        self.active_session_id = saved.id
        return SessionActionResult(
            success=True,
            message=f"Saved session [{saved.id}] '{saved.title}'.",
            session_id=saved.id,
            title=saved.title,
        )

    def load_session_direct(self, session_id: str) -> SessionActionResult:
        if not session_id.strip():
            return SessionActionResult(
                success=False,
                message="Usage: /session-load <id>",
            )
        session = self.session_manager.get_session(session_id.strip())
        if session is None:
            return SessionActionResult(
                success=False,
                message=f"Session '{session_id}' not found.",
            )

        self._clear_pending_state()
        self.turn_count = session.turn_count
        self.active_session_id = session.id
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *session.messages,
        ]
        self.last_retrieved_chunks = []

        if session.summary.strip() and self.features.is_enabled("memory"):
            self.memory_manager.save("session", session.summary)

        self._rebuild_system_prompt()
        return SessionActionResult(
            success=True,
            message=(
                f"Loaded session [{session.id}] '{session.title}' "
                f"({len(session.messages)} messages, {session.turn_count} turns)."
            ),
            session_id=session.id,
            title=session.title,
        )

    def delete_session_direct(self, session_id: str) -> SessionActionResult:
        if not session_id.strip():
            return SessionActionResult(success=False, message="Session ID required.")
        session = self.session_manager.get_session(session_id.strip())
        if session is None:
            return SessionActionResult(
                success=False,
                message=f"Session '{session_id}' not found.",
            )
        if not self.session_manager.delete_session(session.id):
            return SessionActionResult(
                success=False,
                message=f"Failed to delete session '{session_id}'.",
            )
        if self.active_session_id == session.id:
            self.active_session_id = None
        return SessionActionResult(
            success=True,
            message=f"Deleted session '{session.id}'.",
        )

    def run_session_summary(self) -> SessionSummaryResult:
        conversation = self.get_conversation_messages()
        if not conversation:
            return SessionSummaryResult(
                success=False,
                message="Nothing to summarize. Chat first, then run /session-summary.",
            )

        summary = generate_summary(self.runtime, self.messages)
        if not summary.strip():
            return SessionSummaryResult(
                success=False,
                message="Failed to generate session summary.",
            )

        truncated = False
        if self.features.is_enabled("memory"):
            _, truncated = self.memory_manager.save("session", summary)
            self.reload_memory()
        else:
            if self.active_session_id:
                self.session_manager.update_summary(self.active_session_id, summary)
            return SessionSummaryResult(
                success=True,
                message=(
                    "Summary saved to active session file only (memory feature is off). "
                    "Enable memory to inject session.md into prompts."
                ),
                summary=summary,
                truncated=truncated,
            )

        if self.active_session_id:
            self.session_manager.update_summary(self.active_session_id, summary)

        note = " (truncated to session.md limit)" if truncated else ""
        return SessionSummaryResult(
            success=True,
            message=f"Session summary updated{note}. Injected into session.md.",
            summary=summary,
            truncated=truncated,
        )

    def add_user_turn(self, user_input: str) -> list[RetrievedChunk]:
        """Retrieve context, append the user message, and return any sources."""
        self._last_user_input = user_input
        self._last_assistant_reply = ""
        if self.features.is_enabled("memory"):
            self.reload_memory()

        chunks: list[RetrievedChunk] = []
        context_text = ""
        if self.retriever is not None:
            chunks = self.retriever.retrieve(user_input)
            if self.selected_sources is not None and chunks:
                chunks = [c for c in chunks if c.source in self.selected_sources]
            context_text = Retriever.format_context(chunks)

        episodic_context = ""
        self.last_episodic_results = []
        if self.features.is_enabled("memory"):
            self.last_episodic_results = self.search_episodic_memory(
                user_input,
                limit=3,
            )
            if self.last_episodic_results:
                episodic_context = format_episodic_results(
                    self.last_episodic_results
                )

        user_turn = self.prompt_builder.build_user_turn(
            user_input,
            context_text,
            use_rag=self.features.is_enabled("rag"),
            memory=self.memory,
            use_memory=self.features.is_enabled("memory"),
            episodic_context=episodic_context,
        )
        self.messages.append({"role": "user", "content": user_turn})
        self.last_retrieved_chunks = chunks
        return chunks

    def run_ingest(
        self,
        on_progress: Callable[[str, str, int, int], None] | None = None,
    ) -> IngestResult:
        """Index docs/ into ChromaDB and refresh the retriever cache."""
        result = ingest_documents(self.config, self.runtime, on_progress=on_progress)
        if self._retriever is not None:
            self._retriever.reset_collection()
        return result

    def enable_rag(self, sources: list[str] | None = None) -> None:
        """Enable RAG and optionally set document filters."""
        self.set_feature("rag", True)
        if sources is not None:
            self.selected_sources = sources
        elif self.selected_sources is None:
            self.selected_sources = self.get_available_sources()

    def disable_rag(self) -> None:
        """Disable RAG retrieval."""
        self.set_feature("rag", False)

    def toggle_rag(self) -> bool:
        """Toggle RAG on/off and return the new state."""
        return self.features.toggle("rag")

    def set_feature(self, key: str, enabled: bool) -> None:
        """Enable or disable a feature flag."""
        self.features.set_enabled(key, enabled)

    def apply_features(self, changes: dict[str, bool]) -> list[str]:
        """Apply multiple feature changes at once."""
        return self.features.apply_many(changes)

    def get_available_sources(self) -> list[str]:
        """Return available document sources from the vector store."""
        retriever = self._initialize_retriever()
        if retriever is None:
            return []
        return retriever.get_available_sources()

    def set_rag_sources(self, sources: list[str] | None) -> bool:
        """Set which document sources to use for RAG (None = all)."""
        if not self.features.is_enabled("rag"):
            return False
        self.selected_sources = sources
        return True

    def get_rag_stats(self) -> dict[str, int | list[str]]:
        """Return vector store statistics."""
        if self._retriever is not None:
            return self._retriever.get_stats()
        return get_store_stats(self.config)

    def get_rag_status(self) -> dict[str, bool | list[str] | None]:
        """Return current RAG state, enabled status, and selected sources."""
        return {
            "enabled": self.features.is_enabled("rag"),
            "selected_sources": self.selected_sources,
            "available_sources": self.get_available_sources(),
        }

    def run_health_check(self) -> str:
        """Short health summary; never raises."""
        try:
            detail = None
            if self.loaded:
                backend = self.runtime.compute_backend
                detail = f"{backend.label} ({backend.detail})"
            report = run_startup_diagnostics(
                self.config,
                loaded=self.loaded,
                compute_detail=detail,
            )
            return format_health_view(report)
        except Exception as error:  # noqa: BLE001
            return (
                "Health: UNHEALTHY — diagnostics failed\n\n"
                f"  [FAIL] Internal error: {error}\n\n"
                "Run /config to review configuration."
            )

    def run_diagnostics(self) -> str:
        """Full diagnostics report; never raises."""
        try:
            detail = None
            if self.loaded:
                backend = self.runtime.compute_backend
                detail = f"{backend.label} ({backend.detail})"
            report = run_startup_diagnostics(
                self.config,
                loaded=self.loaded,
                compute_detail=detail,
            )
            return format_diagnostics_view(report)
        except Exception as error:  # noqa: BLE001
            return (
                "Diagnostics report:\n\n"
                f"[FAIL] Diagnostics engine error\n"
                f"  {error}\n\n"
                "Check logs/soulforge.log for details."
            )

    def get_config_view(self) -> str:
        """Resolved configuration view for /config."""
        try:
            return format_config_view(self.config)
        except Exception as error:  # noqa: BLE001
            return f"Failed to format config: {error}"

    def _tool_executor(self) -> ToolExecutor:
        return ToolExecutor(
            ToolExecutionContext(
                config=self.config,
                task_manager=self.task_manager,
                turn_count=self.turn_count,
                retriever=self.retriever,
                on_memory_suggestion=self._queue_memory_suggestion_from_tool,
                on_skill_suggestion=self._queue_skill_suggestion_from_tool,
            )
        )

    def _queue_memory_suggestion_from_tool(self, suggestion) -> None:
        self.pending_suggestion = suggestion

    def _queue_skill_suggestion_from_tool(self, suggestion) -> None:
        self.pending_skill_suggestion = suggestion

    def process_assistant_reply(self, raw_text: str) -> ToolTurnResult:
        """Parse tool blocks, update assistant message, run auto tools, queue risky ones."""
        display_text = raw_text.strip()
        if not self.features.is_enabled("tools"):
            self._set_last_assistant_content(display_text)
            return ToolTurnResult(display_text, [], [], None)

        display, calls, parse_error = parse_tool_calls(raw_text)
        if not display.strip():
            display = display_text
        self._set_last_assistant_content(display)

        if parse_error:
            log_tool_event("parse_error", detail=parse_error)
            return ToolTurnResult(display, [], [], parse_error)

        if not calls:
            return ToolTurnResult(display, [], [], None)

        executor = self._tool_executor()
        auto_results: list[ToolResult] = []
        pending: list[PendingToolCall] = []

        for call in calls:
            pending_call = executor.classify(call)
            log_tool_event(
                "proposed",
                call_id=pending_call.call_id,
                name=call.name,
                args=call.args,
                detail=call.rationale,
            )
            if pending_call.requires_approval:
                pending.append(pending_call)
                self.pending_tool_calls.append(pending_call)
            else:
                result = executor.execute(pending_call)
                auto_results.append(result)

        if auto_results:
            self._inject_tool_results(auto_results)

        return ToolTurnResult(display, auto_results, pending, None)

    def _set_last_assistant_content(self, content: str) -> None:
        self._last_assistant_reply = content
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"] = content
        else:
            self.messages.append({"role": "assistant", "content": content})

    def _inject_tool_results(self, results: list[ToolResult]) -> None:
        if not results:
            return
        lines = ["Tool results:"]
        for result in results:
            status = "ok" if result.success else "failed"
            lines.append(f"- {result.name} ({status}): {result.summary()}")
        self.messages.append({"role": "system", "content": "\n".join(lines)})

    def approve_tool_call(self, call_id: str) -> ToolActionResult:
        pending = self._pop_pending_tool(call_id)
        if pending is None:
            return ToolActionResult(False, f"No pending tool call: {call_id}")
        result = self._tool_executor().execute(pending)
        if result.success:
            self._inject_tool_results([result])
            return ToolActionResult(True, result.summary(), result)
        return ToolActionResult(False, result.error or "Tool failed", result)

    def reject_tool_call(self, call_id: str) -> ToolActionResult:
        pending = self._pop_pending_tool(call_id)
        if pending is None:
            return ToolActionResult(False, f"No pending tool call: {call_id}")
        log_tool_event(
            "rejected",
            call_id=call_id,
            name=pending.call.name,
            args=pending.call.args,
        )
        return ToolActionResult(True, f"Rejected tool call {pending.call.name}.")

    def _pop_pending_tool(self, call_id: str) -> PendingToolCall | None:
        for index, pending in enumerate(self.pending_tool_calls):
            if pending.call_id == call_id or pending.call_id.startswith(call_id):
                return self.pending_tool_calls.pop(index)
        return None

    def get_tools_status(self) -> str:
        lines = [format_tools_catalog(self.config)]
        if self.pending_tool_calls:
            lines.append("")
            lines.append(f"Pending approvals: {len(self.pending_tool_calls)}")
            for pending in self.pending_tool_calls:
                lines.append(
                    f"  {pending.call_id}: {pending.call.name} "
                    f"[{pending.risk.value}]"
                )
        return "\n".join(lines)

    def get_tool_log_view(self, limit: int = 20) -> str:
        return read_recent_log(limit)

    def get_tools_menu_data(self) -> dict:
        """Data for the TUI tools workshop modal."""
        tools_cfg = self.config.tools
        return {
            "catalog": format_tools_catalog(self.config),
            "tool_defs": [
                {
                    "name": tool_def.name,
                    "risk": tool_def.risk.value,
                    "description": tool_def.description,
                    "available": is_tool_available(self.config, tool_def.name),
                    "example_args": TOOL_EXAMPLE_ARGS.get(tool_def.name, "{}"),
                }
                for tool_def in list_tool_defs(self.config)
            ],
            "allowlist": list(tools_cfg.shell_allowlist),
            "allow_shell": tools_cfg.allow_shell,
            "allow_write": tools_cfg.allow_write,
            "tools_enabled": self.features.is_enabled("tools"),
            "pending_count": len(self.pending_tool_calls),
        }

    def run_tool_test(self, name: str, args: dict) -> ToolResult:
        """Run a tool immediately for manual testing; bypasses approval modal."""
        name = name.strip()
        try:
            call = ToolCall(name=name, args=args, rationale="manual test")
            pending = PendingToolCall.create(
                call,
                tool_risk(name),
                requires_approval=False,
            )
            log_tool_event(
                "manual_test",
                call_id=pending.call_id,
                name=name,
                args=args,
                detail="user-initiated test",
            )
            result = self._tool_executor().execute(pending)
            result.status = "manual_test"
            return result
        except Exception as error:  # noqa: BLE001
            return ToolResult(
                call_id="",
                name=name,
                success=False,
                error=str(error),
                status="failed",
            )

    def run_agent_workflow(
        self, goal: str, *, on_progress: Callable[[str], None] | None = None
    ) -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.start_run(goal, on_progress=on_progress)

    def get_agents_status(self, run_id: str = "") -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.status(run_id)

    def edit_agent_task(self, task_id: str, new_spec: str = "") -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.edit_task(task_id, new_spec)

    def get_agent_task_input_spec(self, task_id: str) -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.task_input_spec(task_id)

    def approve_agent_checkpoint(self, checkpoint_id: str) -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.approve_checkpoint(checkpoint_id)

    def reject_agent_checkpoint(self, checkpoint_id: str) -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.reject_checkpoint(checkpoint_id)

    def cancel_agent_run(self, run_id: str = "") -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.cancel_run(run_id)

    def agent_planner_profile(self) -> str:
        """Name of the profile the orchestrator plans with."""
        role = self.config.agents.roles.get("orchestrator")
        return role.model_profile if role else self.config.agents.default_profile

    def agent_load_plan(self) -> list[tuple[str, "Path", int]]:
        """Ordered profiles Load Agents will load, deduped by file, with sizes.

        A swap planner evicts everything as it loads, so there is no point
        warming the resident workers ahead of it -- the plan is just the
        planner. Otherwise the residents load first, then the planner. Profiles
        that share a GGUF appear once (they load a single shared instance).
        """
        from pathlib import Path  # local: keep module import surface small

        planner = self.agent_planner_profile()
        prof = self.config.agents.model_profiles.get(planner)
        planner_swaps = prof is not None and prof.residency == "swap"

        names: list[str] = []
        if not planner_swaps and self.config.agents.residency_mode == "hybrid":
            names.extend(
                name
                for name, profile in self.config.agents.model_profiles.items()
                if profile.residency == "resident"
            )
        names.append(planner)

        plan: list[tuple[str, Path, int]] = []
        seen: set[str] = set()
        for name in names:
            path = self.runtime.profile_model_path(name)
            if str(path) in seen:
                continue
            seen.add(str(path))
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            plan.append((name, path, size))
        return plan

    def _load_stats_path(self) -> "Path":
        from pathlib import Path

        return Path(self.config.agents.runs_dir) / ".load_stats.json"

    def estimated_load_bps(self) -> float:
        """Machine's remembered model-load throughput (bytes/sec), 0 if unknown."""
        import json

        try:
            data = json.loads(self._load_stats_path().read_text())
            return max(0.0, float(data.get("bytes_per_sec", 0.0)))
        except Exception:  # noqa: BLE001 - absent/corrupt stats just mean "unknown"
            return 0.0

    def record_load_bps(self, bps: float) -> None:
        """Fold a fresh throughput sample into the persisted estimate (EMA)."""
        import json

        if bps <= 0:
            return
        prev = self.estimated_load_bps()
        blended = bps if prev <= 0 else 0.5 * prev + 0.5 * bps
        try:
            path = self._load_stats_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"bytes_per_sec": blended}))
        except Exception:  # noqa: BLE001 - progress persistence is best-effort
            pass

    def preload_agent_models(
        self, *, on_progress: Callable[[str], None] | None = None
    ) -> str:
        """Load the models a run needs up front instead of during the run.

        The planner's profile is what makes a run appear to hang: it is usually
        a large swap model that only loads once planning starts. Resident
        worker profiles are warmed too, but only when the planner is itself
        resident -- a swap planner evicts everything as it loads, so warming
        them first would just throw the work away (the run re-warms them after
        planning, which is where they belong).
        """
        planner = self.agent_planner_profile()
        prof = self.config.agents.model_profiles.get(planner)
        planner_swaps = prof is not None and prof.residency == "swap"

        lines: list[str] = []
        # Residents go through warm_resident_profiles (not itemized here) so its
        # VRAM fallback is preserved; per-model progress comes from the runtime
        # load listener, which fires for every real load either way.
        if not planner_swaps:
            if on_progress:
                on_progress("Warming resident agent profiles…")
            lines.extend(self.runtime.warm_resident_profiles())

        if on_progress:
            on_progress(f"Loading planner profile '{planner}'…")
        try:
            self.runtime.load_chat_profile(planner)
            lines.append(f"Planner profile '{planner}' loaded.")
        except Exception as error:  # noqa: BLE001
            lines.append(f"Planner profile '{planner}' failed to load: {error}")

        if planner_swaps:
            lines.append(
                f"'{planner}' is a swap profile: it evicts the worker models "
                "while it plans, and they reload afterwards."
            )
        loaded = [s.name for s in self.runtime.profile_statuses() if s.loaded]
        lines.append(f"Loaded profiles: {', '.join(loaded) or 'none'}")
        return "\n".join(lines)

    def resume_agent_run(
        self, run_id: str = "", *, on_progress: Callable[[str], None] | None = None
    ) -> AgentActionResult:
        if not self.features.is_enabled("agents"):
            return AgentActionResult(
                False,
                "Agents are disabled. Run /features agents on or /agents on.",
            )
        return self.agent_manager.resume_run(run_id, on_progress=on_progress)

    def get_agent_model_status(self) -> str:
        lines = [
            f"Agent residency mode: {self.config.agents.residency_mode}",
            "Model profiles:",
        ]
        for status in self.runtime.profile_statuses():
            loaded = "loaded" if status.loaded else "unloaded"
            lines.append(
                f"  {status.name}: {status.residency}, {loaded}, {status.model_path}"
            )
        lines.append(
            "Note: the 70B Orchestrator profile may spill into system memory and "
            "generate more slowly during graph planning."
        )
        return "\n".join(lines)

    def set_tool_permission(self, name: str, enabled: bool) -> str:
        """Grant/revoke a tool permission flag (allowShell/allowWrite/allowNetwork)."""
        attrs = {
            "shell": "allow_shell",
            "allow-shell": "allow_shell",
            "write": "allow_write",
            "allow-write": "allow_write",
            "network": "allow_network",
            "allow-network": "allow_network",
        }
        attr = attrs.get(name.strip().lower())
        if attr is None:
            raise ValueError(
                f"Unknown tool permission '{name}'. Use shell, write, or network."
            )
        setattr(self.config.tools, attr, enabled)
        save_tools(self.config)
        return f"tools.{attr} set to {enabled}. (saved to config.yaml)"

    def add_shell_allowlist_entry(self, command: str) -> str:
        command = command.strip()
        if not command:
            return "Empty command — nothing added."
        allowlist = self.config.tools.shell_allowlist
        if command in allowlist:
            return f"Already on allowlist: {command}"
        allowlist.append(command)
        save_tools(self.config)
        return f"Added to shellAllowlist: {command}"

    def remove_shell_allowlist_entry(self, command: str) -> str:
        command = command.strip()
        allowlist = self.config.tools.shell_allowlist
        if command not in allowlist:
            return f"Not on allowlist: {command}"
        allowlist.remove(command)
        save_tools(self.config)
        return f"Removed from shellAllowlist: {command}"

    def save_onboarding_config(self) -> None:
        """Persist onboarding completion state to config.yaml."""
        save_onboarding(self.config)

    def format_tool_call_preview(self, pending: PendingToolCall) -> str:
        import json

        args_text = json.dumps(pending.call.args, indent=2)
        lines = [
            f"Tool: {pending.call.name}",
            f"Risk: {pending.risk.value}",
            f"ID: {pending.call_id}",
            f"Args:\n{args_text}",
        ]
        if pending.call.rationale:
            lines.append(f"Rationale: {pending.call.rationale}")
        return "\n".join(lines)

    def stream_reply(self) -> Iterator[str]:
        """Yield reply tokens; caller must run process_assistant_reply on full text."""
        stream = self.runtime.create_chat_completion(self.messages, stream=True)
        parts: list[str] = []
        for token in self.runtime.iter_stream_text(stream):
            parts.append(token)
            yield token
        self._pending_raw_reply = "".join(parts).strip()

    def full_reply(self) -> str:
        """Generate a complete (non-streamed) reply; caller runs process_assistant_reply."""
        response = self.runtime.create_chat_completion(self.messages, stream=False)
        return response["choices"][0]["message"]["content"].strip()

    def finalize_assistant_reply(self, raw_text: str) -> ToolTurnResult:
        """Process tool blocks and update conversation state."""
        return self.process_assistant_reply(raw_text)

    @property
    def model_name(self) -> str:
        return self.config.model.chat_model.name

    @property
    def compute_backend(self) -> ComputeBackend:
        return self.runtime.compute_backend

    def active_features(self) -> list[str]:
        return self.features.active_features()

    def features_summary(self) -> str:
        return self.features.summary()
