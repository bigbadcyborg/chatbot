"""FastAPI application wrapping ChatController for the desktop GUI.

Blocking controller calls run in a worker thread so the event loop stays
responsive (the server analog of the TUI's ``@work`` threads). Streaming chat
is delivered over a WebSocket: a worker thread drives the synchronous
``add_user_turn -> stream_reply -> finalize_assistant_reply -> complete_turn``
sequence and pushes JSON frames back onto the event loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from starlette.concurrency import run_in_threadpool

from app.core.chat_controller import ChatController
from app.core.command_router import CommandRouter
from app.server.schemas import (
    AgentLoadStateResponse,
    AgentStartRequest,
    AgentStartResponse,
    AgentStateResponse,
    CommandRequest,
    CommandResponse,
    PingResponse,
    SessionStartRequest,
    SessionStartResponse,
    ShutdownResponse,
    SnapshotResponse,
    TranscribeResponse,
)

_SENTINEL = object()


def _require_token(expected: str):
    """Build a dependency enforcing the shared secret when one is configured."""

    async def _dep(x_soulforge_token: str | None = Header(default=None)) -> None:
        if expected and x_soulforge_token != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing token.")

    return _dep


def _chat_frames(controller: ChatController, text: str, loop, queue: asyncio.Queue) -> None:
    """Run one generation turn on a worker thread, pushing frames to the queue."""

    def emit(frame: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, frame)

    try:
        chunks = controller.add_user_turn(text)
        if controller.features.is_enabled("show_sources") and chunks:
            emit(
                {
                    "type": "sources",
                    "sources": [
                        {"source": c.source, "chunk_index": c.chunk_index}
                        for c in chunks
                    ],
                }
            )
        parts: list[str] = []
        if controller.features.is_enabled("streaming"):
            for token in controller.stream_reply():
                parts.append(token)
                emit({"type": "token", "token": token})
            raw_reply = getattr(controller, "_pending_raw_reply", "") or "".join(parts)
        else:
            raw_reply = controller.full_reply()
            emit({"type": "token", "token": raw_reply})

        tool_turn = controller.finalize_assistant_reply(raw_reply)
        emit({"type": "final", "text": tool_turn.display_text})
        if tool_turn.has_pending:
            emit(
                {
                    "type": "tool",
                    "pending": [
                        controller.format_tool_call_preview(p)
                        for p in tool_turn.pending
                    ],
                }
            )

        review = controller.complete_turn()
        if review.message:
            emit({"type": "review", "text": review.message})
    except Exception as error:  # noqa: BLE001 - surface to the client
        emit({"type": "error", "text": f"Generation error: {error}"})
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)


def run_session_load(controller: ChatController, request, state: dict) -> None:
    """Load the chosen models for a session, updating ``state`` as it goes.

    Runs on a worker thread. ``request`` has chat_model / load_agents /
    load_vision. Best-effort: a vision/agent failure is recorded but does not
    stop the chat model from becoming ready.
    """
    # Surface the exact model being loaded (e.g. "Loading agent model 'creator'")
    # to ping via the shared state, so the GUI shows real per-model progress.
    controller.runtime.set_load_listener(lambda msg: state.__setitem__("stage", msg))
    try:
        state["stage"] = "Loading chat model..."
        chat_model = getattr(request, "chat_model", None)
        current = controller.model_name
        if chat_model and chat_model not in ("", current):
            controller.switch_chat_model(chat_model)
        elif not controller.loaded:
            controller.load()  # skip if the chat model is already loaded

        # RAG and episodic memory embed every prompt, so preload the embedding
        # model too — otherwise the first prompt stalls loading it lazily.
        if controller.features.is_enabled("memory") or controller.features.is_enabled("rag"):
            try:
                controller.runtime.load_embedding_model()
            except Exception as error:  # noqa: BLE001
                state["error"] = f"embedding preload failed: {error}"

        if getattr(request, "load_agents", False):
            try:
                controller.runtime.warm_resident_profiles()
            except Exception as error:  # noqa: BLE001
                state["error"] = f"agent preload failed: {error}"

        if getattr(request, "load_vision", False) and controller.config.vision.enabled:
            try:
                if not controller.runtime.vision_loaded:
                    controller.runtime.preload_vision_model()
                state["vision_loaded"] = True
            except Exception as error:  # noqa: BLE001
                state["error"] = f"vision preload failed: {error}"
    except Exception as error:  # noqa: BLE001
        state["error"] = f"model load failed: {error}"
    finally:
        controller.runtime.set_load_listener(None)
        state["stage"] = "ready"
        state["loading"] = False


def _request_stop(controller: ChatController) -> None:
    """Release the models, then signal uvicorn to shut down.

    Runs on a background thread so the HTTP response reaches the GUI before the
    process goes away. Tests patch this out — it terminates the interpreter.
    """

    def stop() -> None:
        import os
        import signal
        import time

        time.sleep(0.4)  # let the response flush first
        try:
            controller.runtime.unload_chat_model()
            controller.runtime.unload_vision_model()
        except Exception:  # noqa: BLE001 - going down regardless
            pass
        os.kill(os.getpid(), signal.SIGTERM)  # uvicorn handles this gracefully

    threading.Thread(target=stop, daemon=True).start()


def create_app(controller: ChatController, transcriber=None) -> FastAPI:
    """Build the FastAPI app bound to a ready ChatController.

    ``transcriber`` is injectable (tests pass a stub); when omitted a lazy
    faster-whisper ``Transcriber`` is created — it loads no model until the
    first /transcribe call.
    """
    app = FastAPI(title="SoulForge API", version="1.0")
    router = CommandRouter(controller)
    auth = _require_token(controller.config.server.auth_token)

    if transcriber is None:
        from app.server.transcribe import Transcriber

        transcriber = Transcriber(controller.config.transcription)

    # Shared session-load state (updated by the loader thread, read by ping).
    session_state: dict = {"stage": "idle", "vision_loaded": False, "loading": False}
    # Agent runs are long (model swaps + many LLM calls), so they run in a
    # background thread and the GUI polls /api/agents/state — a single blocking
    # request would exceed any sane HTTP timeout.
    agent_state: dict = {"running": False, "stage": "", "result": ""}
    # Load Agents is also long (a 40 GB planner load), so it runs in a thread
    # and the GUI polls /api/agents/load-state to drive a progress bar. llama
    # gives no sub-load progress, so the bar advances per model completed; the
    # ETA comes from throughput remembered across loads.
    load_state: dict = {
        "loading": False,
        "done": False,
        "current": "",
        "loaded_bytes": 0,
        "total_bytes": 0,
        "started_at": 0.0,
        "message": "",
    }

    def _run_load_job() -> None:
        plan = controller.agent_load_plan()
        already = controller.runtime.loaded_profile_names()
        sizes = {name: size for name, _path, size in plan}
        # Only count what will actually load; residents already resident are free.
        total = sum(size for name, _p, size in plan if name not in already)

        # A single mutable holder so the listener closure can finalize the
        # previous model's bytes when the next one starts loading.
        cur = {"name": "", "size": 0}

        def finalize_current() -> None:
            if cur["name"]:
                load_state["loaded_bytes"] = min(
                    total, load_state["loaded_bytes"] + cur["size"]
                )
                cur["name"], cur["size"] = "", 0

        def on_load(message: str) -> None:
            # Messages: "Loading agent model 'X'...", "Loading chat model...".
            name = ""
            if "agent model '" in message:
                name = message.split("agent model '", 1)[1].split("'", 1)[0]
            elif "chat model" in message:
                name = "default"
            else:
                return  # embedding model etc. -- not part of the agent load
            finalize_current()  # the model that was loading has finished
            cur["name"] = name
            cur["size"] = sizes.get(name, 0)
            load_state["current"] = name

        previous_listener = controller.runtime._load_listener
        controller.runtime.set_load_listener(on_load)
        started = time.monotonic()
        load_state.update(
            {
                "loading": True,
                "done": False,
                "current": "",
                "loaded_bytes": 0,
                "total_bytes": total,
                "started_at": started,
                "message": "Loading agent models…",
            }
        )
        try:
            summary = controller.preload_agent_models()
            finalize_current()  # the last model has no "next" event to close it
            load_state["message"] = summary
            elapsed = time.monotonic() - started
            if load_state["loaded_bytes"] > 0 and elapsed > 0:
                controller.record_load_bps(load_state["loaded_bytes"] / elapsed)
        except Exception as error:  # noqa: BLE001
            load_state["message"] = f"Load failed: {error}"
        finally:
            controller.runtime.set_load_listener(previous_listener)
            load_state["loaded_bytes"] = load_state["total_bytes"]
            load_state["current"] = ""
            load_state["loading"] = False
            load_state["done"] = True

    def _run_agent_job(kind: str, goal: str, run_id: str) -> None:
        def progress(line: str) -> None:
            agent_state["stage"] = line

        try:
            agent_state.update({"running": True, "stage": "starting", "result": ""})
            if kind == "resume":
                result = controller.resume_agent_run(run_id, on_progress=progress)
            else:
                result = controller.run_agent_workflow(goal, on_progress=progress)
            agent_state["result"] = result.message
        except Exception as error:  # noqa: BLE001
            agent_state["result"] = f"Agent run failed: {error}"
        finally:
            agent_state["running"] = False
            agent_state["stage"] = "done"

    @app.post(
        "/api/agents/start",
        response_model=AgentStartResponse,
        dependencies=[Depends(auth)],
    )
    async def agents_start(request: AgentStartRequest) -> AgentStartResponse:
        if agent_state["running"]:
            return AgentStartResponse(started=False, message="An agent run is already in progress.")
        if not request.goal.strip():
            return AgentStartResponse(started=False, message="Goal is required.")
        threading.Thread(
            target=_run_agent_job, args=("run", request.goal, ""), daemon=True
        ).start()
        return AgentStartResponse(started=True, message="Agent run started.")

    @app.post(
        "/api/agents/resume",
        response_model=AgentStartResponse,
        dependencies=[Depends(auth)],
    )
    async def agents_resume(request: AgentStartRequest) -> AgentStartResponse:
        if agent_state["running"]:
            return AgentStartResponse(started=False, message="An agent run is already in progress.")
        threading.Thread(
            target=_run_agent_job, args=("resume", "", request.run_id), daemon=True
        ).start()
        return AgentStartResponse(started=True, message="Resuming agent run.")

    @app.get(
        "/api/agents/state",
        response_model=AgentStateResponse,
        dependencies=[Depends(auth)],
    )
    async def agents_state() -> AgentStateResponse:
        # Read the flags *before* the run record. A client stops polling when it
        # sees running=False, so that snapshot must carry the finished run: read
        # the other way round and a run completing in between reports "done"
        # alongside a record written before the final answer landed.
        running = agent_state["running"]
        stage = agent_state["stage"]
        result = agent_state["result"]
        data = await run_in_threadpool(controller.agents_data, "")
        return AgentStateResponse(
            running=running, stage=stage, result=result, data=data
        )

    @app.post(
        "/api/agents/load",
        response_model=AgentStartResponse,
        dependencies=[Depends(auth)],
    )
    async def agents_load() -> AgentStartResponse:
        if load_state["loading"]:
            return AgentStartResponse(started=False, message="Already loading.")
        if agent_state["running"]:
            return AgentStartResponse(
                started=False, message="An agent run is in progress."
            )
        # Flip the flags synchronously so a poll landing before the thread runs
        # can't read a previous load's done=True and stop instantly.
        load_state.update({"loading": True, "done": False})
        threading.Thread(target=_run_load_job, daemon=True).start()
        return AgentStartResponse(started=True, message="Loading agent models.")

    @app.get(
        "/api/agents/load-state",
        response_model=AgentLoadStateResponse,
        dependencies=[Depends(auth)],
    )
    async def agents_load_state() -> AgentLoadStateResponse:
        loaded = load_state["loaded_bytes"]
        total = load_state["total_bytes"]
        started = load_state["started_at"]
        elapsed = (time.monotonic() - started) if started else 0.0
        remaining = max(0, total - loaded)
        # ETA from the machine's remembered load throughput. The live rate is
        # unusable here: bytes land in one lump per model, so a single large
        # load reads as 0 B/s until it finishes.
        bps = await run_in_threadpool(controller.estimated_load_bps)
        eta = (remaining / bps) if (bps > 0 and load_state["loading"]) else -1.0
        return AgentLoadStateResponse(
            loading=load_state["loading"],
            done=load_state["done"],
            current=load_state["current"],
            loaded_bytes=loaded,
            total_bytes=total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            message=load_state["message"],
        )

    @app.post(
        "/api/shutdown",
        response_model=ShutdownResponse,
        dependencies=[Depends(auth)],
    )
    async def shutdown() -> ShutdownResponse:
        """Release the models and stop the server, replying before it exits."""
        _request_stop(controller)
        return ShutdownResponse(stopping=True, message="Server shutting down.")

    @app.get("/api/ping", response_model=PingResponse)
    async def ping() -> PingResponse:
        stage = session_state["stage"]
        if stage in ("idle", "ready"):
            stage = "ready" if controller.loaded else "idle"
        return PingResponse(
            status="ok",
            model=controller.model_name,
            ready=controller.loaded,
            compute_backend=str(controller.compute_backend),
            stage=stage,
            vision_loaded=session_state["vision_loaded"] or controller.runtime.vision_loaded,
            loading=session_state["loading"],
        )

    @app.post(
        "/api/session/start",
        response_model=SessionStartResponse,
        dependencies=[Depends(auth)],
    )
    async def session_start(request: SessionStartRequest) -> SessionStartResponse:
        if session_state["loading"]:
            return SessionStartResponse(started=False, message="Already loading.")
        session_state.update({"loading": True, "stage": "starting", "error": None})
        threading.Thread(
            target=run_session_load,
            args=(controller, request, session_state),
            daemon=True,
        ).start()
        return SessionStartResponse(started=True, message="Loading started.")

    @app.get("/api/commands", dependencies=[Depends(auth)])
    async def commands() -> dict[str, list[str]]:
        return {"commands": router.command_names()}

    @app.post("/api/command", response_model=CommandResponse, dependencies=[Depends(auth)])
    async def command(request: CommandRequest) -> CommandResponse:
        # Router calls into blocking ChatController methods; keep them off-loop.
        result = await run_in_threadpool(router.dispatch, request.name, request.args)
        return CommandResponse(**result.to_dict())

    @app.post("/api/snapshot", response_model=SnapshotResponse, dependencies=[Depends(auth)])
    async def snapshot(
        image: UploadFile = File(...),
        prompt: str = Form(default=""),
        inject: bool = Form(default=True),
    ) -> SnapshotResponse:
        if not controller.config.vision.enabled:
            raise HTTPException(
                status_code=400,
                detail="Vision model not configured (set vision.modelPath).",
            )
        data = await image.read()

        def run() -> str:
            answer = controller.runtime.create_vision_completion(data, prompt)
            if inject:
                # Keep the model conversation coherent for follow-up questions.
                label = prompt or "Describe this image."
                controller.messages.append(
                    {"role": "user", "content": f"[Screen snapshot] {label}"}
                )
                controller.messages.append({"role": "assistant", "content": answer})
            return answer

        answer = await run_in_threadpool(run)
        return SnapshotResponse(text=answer)

    @app.post(
        "/api/transcribe",
        response_model=TranscribeResponse,
        dependencies=[Depends(auth)],
    )
    async def transcribe(
        audio: UploadFile = File(...),
        language: str = Form(default=""),
    ) -> TranscribeResponse:
        data = await audio.read()
        text = await run_in_threadpool(transcriber.transcribe_wav, data, language)
        return TranscribeResponse(text=text)

    @app.post("/api/rag/upload", response_model=CommandResponse, dependencies=[Depends(auth)])
    async def rag_upload(document: UploadFile = File(...)) -> CommandResponse:
        data = await document.read()
        text = await run_in_threadpool(
            controller.save_uploaded_doc, document.filename or "document", data
        )
        return CommandResponse(kind="message", text=text, success=True)

    @app.websocket("/ws/chat")
    async def chat(websocket: WebSocket) -> None:
        token = controller.config.server.auth_token
        if token and websocket.query_params.get("token") != token:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        try:
            while True:
                payload = await websocket.receive_json()
                text = str(payload.get("message", "")).strip()
                if not text:
                    await websocket.send_json({"type": "error", "text": "Empty message."})
                    continue
                if not controller.loaded or session_state["loading"]:
                    # A background load holds the model lock; refuse rather than
                    # hang the prompt until it finishes.
                    await websocket.send_json(
                        {"type": "error", "text": "Models still loading — please wait."}
                    )
                    continue
                queue: asyncio.Queue = asyncio.Queue()
                threading.Thread(
                    target=_chat_frames,
                    args=(controller, text, loop, queue),
                    daemon=True,
                ).start()
                while True:
                    frame = await queue.get()
                    if frame is _SENTINEL:
                        break
                    await websocket.send_json(frame)
                await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            return

    return app
