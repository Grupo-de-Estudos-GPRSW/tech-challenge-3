"""Lazy loader and session layer around the LangGraph pipeline in `src/`.

Importing `src.graph` is expensive: it logs into the Hugging Face Hub, downloads /
loads the fine-tuned 7B model and builds the FAISS index with OpenAI embeddings.
The web server therefore never imports it at module level -- it kicks off
`start_loading()` in a background thread and reports progress through `get_status()`.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Human readable labels for every node of the graph, used to narrate progress in the UI.
NODE_LABELS: Dict[str, str] = {
    "parse_and_validate_input": "Parsing and validating input",
    "request_patient_id": "Asking for a patient identifier",
    "fetch_ehr_context": "Fetching the patient EHR context",
    "retrieve_protocols": "Retrieving clinical protocols",
    "generate_llm_response": "Generating the response with the fine-tuned model",
    "guardrail_evaluator": "Evaluating safety guardrails",
    "human_validation_gate": "Holding for physician sign-off",
    "audit_logger": "Writing the audit trail",
}


# --------------------------------------------------------------------------- #
# Graph loading
# --------------------------------------------------------------------------- #

class _GraphLoader:
    """Loads `src.graph.app` once, in the background, and exposes its status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._app: Any = None
        self._state = "idle"          # idle | loading | ready | error
        self._detail = "Not started"
        self._error: Optional[str] = None

    # -- status ------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"state": self._state, "detail": self._detail, "error": self._error}

    def _set(self, state: str, detail: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._state, self._detail, self._error = state, detail, error

    # -- loading ----------------------------------------------------------- #
    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._load, name="graph-loader", daemon=True)
            self._state, self._detail = "loading", "Starting up"
        self._thread.start()

    def _load(self) -> None:
        try:
            self._set("loading", "Building the protocol vector store (OpenAI embeddings)")
            import src.vectorstore  # noqa: F401  -- fail fast on a missing OPENAI_API_KEY

            self._set("loading", "Loading the fine-tuned model from the Hugging Face Hub")
            import src.model_loading  # noqa: F401

            self._set("loading", "Compiling the LangGraph workflow")
            from src.graph import app

            self._app = app
            self._set("ready", "Pipeline ready")
        except BaseException as exc:  # noqa: BLE001 -- surfaced verbatim in the UI
            self._set("error", f"{type(exc).__name__}: {exc}", traceback.format_exc())

    def get(self) -> Any:
        if self._app is None:
            raise RuntimeError(self.status()["detail"])
        return self._app


loader = _GraphLoader()


def start_loading() -> None:
    loader.start()


def get_status() -> Dict[str, Any]:
    return loader.status()


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

@dataclass
class Session:
    """One browser conversation.

    `GraphState` is single-turn, so the only thing worth carrying across turns is
    the patient identifier: once the physician names a patient, follow-up questions
    keep resolving against the same record.
    """

    session_id: str
    patient_id: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)


_sessions: Dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str) -> Session:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            _sessions[session_id] = session
        return session


def reset_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)


# --------------------------------------------------------------------------- #
# Running one turn
# --------------------------------------------------------------------------- #

def initial_state(user_query: str, patient_id: Optional[str]) -> Dict[str, Any]:
    return {
        "user_query": user_query,
        "patient_id": patient_id,
        "missing_patient_id": False,
        "patient_context": {},
        "retrieved_docs": [],
        "llm_output": "",
        "requires_human_approval": False,
        "audit_payload": {},
    }


_SENTINEL = object()


def run_turn(session: Session, user_query: str) -> Iterator[Dict[str, Any]]:
    """Run one graph execution, yielding `{"type": ...}` events as nodes complete.

    Event types: `node` (a node finished), `result` (final answer), `error`.
    """
    graph = loader.get()
    events: "queue.Queue[Any]" = queue.Queue()

    # A brand new patient id in this turn's query overrides the session one; the
    # graph's own regex does the extraction, so we only pass the remembered id.
    state = initial_state(user_query, session.patient_id)
    final_state: Dict[str, Any] = dict(state)

    def worker() -> None:
        try:
            for update in graph.stream(state, stream_mode="updates"):
                for node_name, node_output in update.items():
                    if isinstance(node_output, dict):
                        final_state.update(node_output)
                    events.put({"type": "node", "node": node_name,
                                "label": NODE_LABELS.get(node_name, node_name)})
        except BaseException as exc:  # noqa: BLE001
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()})
        finally:
            events.put(_SENTINEL)

    threading.Thread(target=worker, name="graph-turn", daemon=True).start()

    failed = False
    while True:
        event = events.get()
        if event is _SENTINEL:
            break
        if event.get("type") == "error":
            failed = True
        yield event

    if failed:
        return

    if final_state.get("patient_id"):
        session.patient_id = final_state["patient_id"]

    audit = final_state.get("audit_payload", {}) or {}
    yield {
        "type": "result",
        "answer": final_state.get("llm_output", ""),
        "patient_id": final_state.get("patient_id"),
        "missing_patient_id": bool(final_state.get("missing_patient_id")),
        "patient_context": final_state.get("patient_context", {}) or {},
        "retrieved_docs": final_state.get("retrieved_docs", []) or [],
        "requires_human_approval": bool(final_state.get("requires_human_approval")),
        "original_output": audit.get("original_llm_output_for_review"),
        "log_filepath": audit.get("log_filepath"),
    }
