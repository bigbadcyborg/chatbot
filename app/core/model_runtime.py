"""Model runtime: loads and serves the local GGUF chat and embedding models.

This module wraps ``llama_cpp.Llama`` so the rest of the app never touches the
backend directly. The heavy ``llama_cpp`` import is deferred until a model is
actually loaded, which keeps config/prompt logic importable in lightweight
environments (tests, tooling) without the native dependency.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

from app.core.compute_backend import UNKNOWN, ComputeBackend, detect_compute_backend
from app.core.config import AgentModelProfileConfig, AppConfig
from app.utils.audit_logger import append_audit_event

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_cpp import Llama


@dataclass
class RuntimeProfileStatus:
    name: str
    loaded: bool
    residency: str
    model_path: str


class ModelRuntime:
    """Owns the chat model and the optional embedding model."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._chat: "Llama | None" = None
        self._chat_profiles: dict[str, "Llama"] = {}
        self._profile_paths: dict[str, Path] = {}
        self._embedder: "Llama | None" = None
        self._vision: "Llama | None" = None
        self._compute_backend: ComputeBackend = UNKNOWN
        self._lock = threading.RLock()
        self._active_profile: str = ""
        self._resident_fallback: bool = False
        self._profile_errors: list[str] = []
        self._load_listener: Callable[[str], None] | None = None

    def set_load_listener(self, listener: Callable[[str], None] | None) -> None:
        """Register a callback notified with a message when a model starts loading."""
        self._load_listener = listener

    def _notify_load(self, message: str) -> None:
        print(message)
        if self._load_listener is not None:
            try:
                self._load_listener(message)
            except Exception:  # noqa: BLE001 - progress is best-effort
                pass

    @property
    def compute_backend(self) -> ComputeBackend:
        return self._compute_backend

    def load_chat_model(self) -> "Llama":
        """Load the GGUF chat model, raising a clear error if it is missing."""
        with self._lock:
            chat = self._load_chat_profile_unlocked("default")
            self._chat = chat
            return chat

    def unload_chat_model(self) -> None:
        """Release all loaded chat profiles and reset compute backend detection."""
        with self._lock:
            self._chat = None
            self._chat_profiles.clear()
            self._profile_paths.clear()
            self._active_profile = ""
            self._compute_backend = UNKNOWN
        gc.collect()

    def reload_chat_model(self) -> "Llama":
        """Unload and load the chat model from the current config path."""
        with self._lock:
            self._chat = None
            self._chat_profiles.clear()
            self._profile_paths.clear()
            self._active_profile = ""
            self._compute_backend = UNKNOWN
        gc.collect()
        with self._lock:
            chat = self._load_chat_profile_unlocked("default", force=True)
            self._chat = chat
            return chat

    def unload_chat_profile(self, profile_name: str) -> None:
        """Release a single named chat profile if it is loaded."""
        key = self._normalize_profile_name(profile_name)
        with self._lock:
            self._chat_profiles.pop(key, None)
            self._profile_paths.pop(key, None)
            if key == "default":
                self._chat = None
            if self._active_profile == key:
                self._active_profile = ""
        gc.collect()

    def warm_resident_profiles(self) -> list[str]:
        """Best-effort load of resident agent profiles.

        Returns warning strings. Resident loading is opportunistic by design:
        if the local VRAM budget cannot hold the configured resident models, the
        runtime unloads partial state and falls back to sequential hot-swapping.
        """
        warnings: list[str] = []
        if self.config.agents.residency_mode != "hybrid":
            self._resident_fallback = True
            return warnings

        resident_names = [
            name
            for name, profile in self.config.agents.model_profiles.items()
            if profile.residency == "resident"
        ]
        for name in resident_names:
            try:
                with self._lock:
                    self._load_chat_profile_unlocked(name)
            except Exception as error:  # noqa: BLE001
                warnings.append(
                    f"Resident profile '{name}' failed to load; using sequential fallback: {error}"
                )
                with self._lock:
                    self._chat_profiles.clear()
                    self._profile_paths.clear()
                    self._chat = None
                    self._resident_fallback = True
                    self._profile_errors.extend(warnings)
                gc.collect()
                break
        return warnings

    def load_chat_profile(self, profile_name: str) -> None:
        """Load one named profile now, so a later call does not pay for it."""
        with self._lock:
            self._load_chat_profile_unlocked(profile_name)

    def profile_model_path(self, profile_name: str) -> Path:
        """Public: the GGUF a profile loads from (its override, or the default)."""
        return self._profile_model_path(self._normalize_profile_name(profile_name))

    def loaded_profile_names(self) -> set[str]:
        """Names of profiles with a model currently resident."""
        with self._lock:
            return set(self._chat_profiles)

    def profile_statuses(self) -> list[RuntimeProfileStatus]:
        """Return loaded/unloaded state for configured runtime profiles."""
        statuses = [
            RuntimeProfileStatus(
                name="default",
                loaded="default" in self._chat_profiles,
                residency="resident",
                model_path=str(self.config.model.chat_model),
            )
        ]
        for name, profile in self.config.agents.model_profiles.items():
            statuses.append(
                RuntimeProfileStatus(
                    name=name,
                    loaded=name in self._chat_profiles,
                    residency=profile.residency,
                    model_path=str(self._profile_model_path(name)),
                )
            )
        return statuses

    def load_embedding_model(self) -> "Llama":
        """Load the GGUF embedding model, raising a clear error if missing."""
        with self._lock:
            return self._load_embedding_model_unlocked()

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a piece of text."""
        with self._lock:
            embedder = self._load_embedding_model_unlocked()
            result = embedder.create_embedding(text)
            return result["data"][0]["embedding"]

    # -- vision (multimodal) --------------------------------------------

    def unload_vision_model(self) -> None:
        """Release the loaded vision model, if any."""
        with self._lock:
            self._vision = None
        gc.collect()

    def preload_vision_model(self) -> None:
        """Load the vision model now so the first snapshot is fast."""
        with self._lock:
            self._load_vision_unlocked()

    @property
    def vision_loaded(self) -> bool:
        return self._vision is not None

    def create_vision_completion(self, image_bytes: bytes, prompt: str) -> str:
        """Describe/answer about an image using the configured vision model.

        Raises ``RuntimeError`` if no vision model is configured. The image is
        passed as a base64 data URI content part (the OpenAI-vision message
        shape llama-cpp's chat handlers expect).
        """
        vision_cfg = self.config.vision
        if not vision_cfg.enabled:
            raise RuntimeError("No vision model configured (set vision.modelPath).")

        import base64

        data_uri = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe this image."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        with self._lock:
            vision = self._load_vision_unlocked()
            response = vision.create_chat_completion(
                messages=messages,
                max_tokens=vision_cfg.max_tokens,
            )
        return response["choices"][0]["message"]["content"].strip()

    def _load_vision_unlocked(self) -> "Llama":
        if self._vision is not None:
            return self._vision

        vision_cfg = self.config.vision
        model_path = vision_cfg.model
        mmproj_path = vision_cfg.mmproj
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                f"Vision model not found: {model_path}. Check vision.modelPath."
            )
        if mmproj_path is None or not mmproj_path.exists():
            raise FileNotFoundError(
                f"Vision mmproj not found: {mmproj_path}. Check vision.mmprojPath."
            )

        # Optionally free the chat model's VRAM first (it reloads on next use).
        if vision_cfg.evict_chat:
            self._chat = None
            self._chat_profiles.clear()
            self._profile_paths.clear()
            self._active_profile = ""
            gc.collect()

        from llama_cpp import Llama

        handler = self._build_vision_handler(str(mmproj_path))
        self._notify_load(f"Loading vision model '{model_path.name}'...")
        self._vision = Llama(
            model_path=str(model_path),
            chat_handler=handler,
            n_ctx=vision_cfg.context_size,
            n_gpu_layers=self.config.model.gpu_layers,
            verbose=False,
        )
        return self._vision

    def _build_vision_handler(self, mmproj_path: str):
        """Map the config handler name to a llama-cpp chat handler instance."""
        from llama_cpp import llama_chat_format

        handlers = {
            "llava-1-5": "Llava15ChatHandler",
            "llava-1-6": "Llava16ChatHandler",
            "moondream": "MoondreamChatHandler",
            "qwen2.5-vl": "Qwen25VLChatHandler",
        }
        class_name = handlers.get(self.config.vision.chat_handler, "Llava15ChatHandler")
        handler_cls = getattr(
            llama_chat_format, class_name, llama_chat_format.Llava15ChatHandler
        )
        return handler_cls(clip_model_path=mmproj_path)

    def _completion_params(self, **overrides: Any) -> dict[str, Any]:
        gen = self.config.generation
        params: dict[str, Any] = {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "repeat_penalty": gen.repeat_penalty,
            "max_tokens": gen.max_tokens,
            "stop": gen.stop,
        }
        params.update(overrides)
        return params

    def _normalize_profile_name(self, profile_name: str | None) -> str:
        key = (profile_name or "default").strip()
        return key or "default"

    def _profile_config(self, profile_name: str) -> AgentModelProfileConfig | None:
        if profile_name == "default":
            return None
        return self.config.agents.model_profiles.get(profile_name)

    def _profile_model_path(self, profile_name: str) -> Path:
        profile = self._profile_config(profile_name)
        if profile is not None and profile.chat_model is not None:
            return profile.chat_model
        return self.config.model.chat_model

    def _profile_chat_format(self, profile_name: str) -> str:
        """Chat format a profile loads with (profile override or model default)."""
        profile = self._profile_config(profile_name)
        if profile is not None and profile.chat_format:
            return profile.chat_format
        return self.config.model.chat_format

    def _profile_residency(self, profile_name: str) -> str:
        profile = self._profile_config(profile_name)
        if profile is None:
            return "resident"
        return profile.residency

    def _profile_completion_overrides(self, profile_name: str) -> dict[str, Any]:
        profile = self._profile_config(profile_name)
        if profile is None:
            return {}
        overrides: dict[str, Any] = {}
        if profile.temperature is not None:
            overrides["temperature"] = profile.temperature
        if profile.top_p is not None:
            overrides["top_p"] = profile.top_p
        if profile.repeat_penalty is not None:
            overrides["repeat_penalty"] = profile.repeat_penalty
        if profile.max_tokens is not None:
            overrides["max_tokens"] = profile.max_tokens
        return overrides

    def _load_chat_profile_unlocked(
        self,
        profile_name: str,
        *,
        force: bool = False,
    ) -> "Llama":
        key = self._normalize_profile_name(profile_name)
        model_path = self._profile_model_path(key)
        loaded = self._chat_profiles.get(key)
        if loaded is not None and not force and self._profile_paths.get(key) == model_path:
            self._active_profile = key
            return loaded

        residency = self._profile_residency(key)
        if residency not in ("resident", "swap"):
            raise ValueError(
                f"Invalid residency for profile '{key}': {residency}. "
                "Use 'resident' or 'swap'."
            )

        if force:
            self._chat_profiles.pop(key, None)
            self._profile_paths.pop(key, None)

        if residency == "swap" or self._resident_fallback or self.config.agents.residency_mode == "sequential":
            for loaded_key in list(self._chat_profiles):
                if loaded_key != key:
                    self._chat_profiles.pop(loaded_key, None)
                    self._profile_paths.pop(loaded_key, None)
            self._chat = None
            gc.collect()
        else:
            # A swap profile only borrows memory while it runs: release it as
            # soon as any other profile loads, or the 70B orchestrator would
            # stay resident alongside the worker models and blow the VRAM
            # budget.
            evicted = False
            for loaded_key in list(self._chat_profiles):
                if loaded_key != key and self._profile_residency(loaded_key) == "swap":
                    self._chat_profiles.pop(loaded_key, None)
                    self._profile_paths.pop(loaded_key, None)
                    evicted = True
            if evicted:
                gc.collect()

        from llama_cpp import Llama

        if not model_path.exists():
            raise FileNotFoundError(
                f"Chat model not found for profile '{key}': {model_path}. "
                "Check model.chatModelPath or agents.modelProfiles in config.yaml."
            )

        # Several agent roles usually point at the same GGUF. Reuse the loaded
        # instance instead of loading a second copy — otherwise N roles means N
        # copies of the same weights and the VRAM budget blows up.
        wanted_format = self._profile_chat_format(key)
        for loaded_key, loaded_model in self._chat_profiles.items():
            if (
                self._profile_paths.get(loaded_key) == model_path
                and self._profile_chat_format(loaded_key) == wanted_format
            ):
                self._chat_profiles[key] = loaded_model
                self._profile_paths[key] = model_path
                self._active_profile = key
                if key == "default":
                    self._chat = loaded_model
                return loaded_model

        label = "chat model" if key == "default" else f"agent model '{key}'"
        self._notify_load(f"Loading {label}...")
        chat_format = wanted_format
        chat = Llama(
            model_path=str(model_path),
            n_ctx=self.config.model.context_size,
            n_gpu_layers=self.config.model.gpu_layers,
            n_threads=self.config.model.threads,
            chat_format=chat_format,
            verbose=False,
        )
        self._chat_profiles[key] = chat
        self._profile_paths[key] = model_path
        self._active_profile = key
        if key == "default":
            self._chat = chat
        self._compute_backend = detect_compute_backend(self.config)
        return chat

    def _load_embedding_model_unlocked(self) -> "Llama":
        if self._embedder is not None:
            return self._embedder

        from llama_cpp import Llama

        model_path = self.config.model.embedding_model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model not found: {model_path}. "
                "Check model.embeddingModelPath in config.yaml, "
                "or disable RAG in config.yaml (features.rag: false)."
            )

        self._notify_load("Loading embedding model...")
        self._embedder = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=2048,
            n_gpu_layers=self.config.model.gpu_layers,
            verbose=False,
        )
        return self._embedder

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        **overrides: Any,
    ) -> Any:
        """Run a chat completion using configured generation defaults.

        Any keyword in ``overrides`` takes precedence over the config values.
        Returns the raw llama_cpp response, or a streaming iterator when
        ``stream`` is True.
        """
        return self.create_chat_completion_for_profile(
            "default",
            messages,
            stream=stream,
            **overrides,
        )

    def create_chat_completion_for_profile(
        self,
        profile_name: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        **overrides: Any,
    ) -> Any:
        """Run a chat completion against a named runtime profile."""
        profile_overrides = self._profile_completion_overrides(profile_name)
        profile_overrides.update(overrides)
        params = self._completion_params(**profile_overrides)
        if stream:
            return self._stream_chat_completion_for_profile(
                profile_name,
                messages,
                **profile_overrides,
            )

        with self._lock:
            chat = self._load_chat_profile_unlocked(profile_name)
            try:
                response = chat.create_chat_completion(
                    messages=messages,
                    stream=False,
                    **params,
                )
            except Exception as error:
                append_audit_event(
                    self.config,
                    messages=messages,
                    params=params,
                    stream=False,
                    error=str(error),
                )
                raise

            text = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            append_audit_event(
                self.config,
                messages=messages,
                params=params,
                response_text=text,
                stream=False,
            )
            return response

    def _stream_chat_completion(
        self,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> Iterator[dict[str, Any]]:
        return self._stream_chat_completion_for_profile(
            "default",
            messages,
            **overrides,
        )

    def _stream_chat_completion_for_profile(
        self,
        profile_name: str,
        messages: list[dict[str, str]],
        **overrides: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream tokens while holding the runtime lock for the full decode."""
        params = self._completion_params(**overrides)
        with self._lock:
            chat = self._load_chat_profile_unlocked(profile_name)
            response_parts: list[str] = []
            try:
                stream = chat.create_chat_completion(
                    messages=messages,
                    stream=True,
                    **params,
                )
                for chunk in stream:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        response_parts.append(content)
                    yield chunk
            except Exception as error:
                append_audit_event(
                    self.config,
                    messages=messages,
                    params=params,
                    response_text="".join(response_parts),
                    stream=True,
                    error=str(error),
                )
                raise
            append_audit_event(
                self.config,
                messages=messages,
                params=params,
                response_text="".join(response_parts),
                stream=True,
            )

    @staticmethod
    def iter_stream_text(stream: Iterator[dict[str, Any]]) -> Iterator[str]:
        """Yield incremental text from a streaming chat completion."""
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content
