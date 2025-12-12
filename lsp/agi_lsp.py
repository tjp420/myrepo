"""AGI Language Server (LSP) — small pygls server that proxies completions/hover/code actions
to the AGI VSCode adapter (`vscode/agi_vscode_adapter.py`).

This server is intentionally minimal: it forwards document text and cursor position
to the adapter's `/assist` endpoint and uses the returned suggestion to form
completion items and hover content. It works with editors that support LSP (VS Code, Windsurf, etc.).
"""
import os
import json
import logging
from typing import List

import httpx
import time
import random
from typing import Optional
import threading
import traceback
import uuid
from typing import Dict, Any

# Robust import for pygls LanguageServer: try common locations and provide a
# lightweight stub if pygls is not present or its API differs. This keeps the
# module importable during tests without forcing a specific pygls install.
try:
    from pygls.server import LanguageServer
except Exception:
    try:
        import pygls.server as _pg_server
        LanguageServer = getattr(_pg_server, "LanguageServer", getattr(_pg_server, "Server", None))
        # If the attribute isn't present, trigger the fallback to our stub
        if LanguageServer is None:
            raise ImportError("pygls.server has no LanguageServer/Server")
    except Exception:
        class LanguageServer:
            def __init__(self, *args, **kwargs):
                # Minimal workspace placeholder; tests may monkeypatch this.
                class _WorkspaceStub:
                    def __init__(self):
                        self._docs = {}
                    def get_document(self, uri):
                        # Return a minimal document-like object
                        return type('Doc', (), {
                            'source': '',
                            'uri': uri,
                            'path': None,
                            'language_id': 'plaintext'
                        })

                self.workspace = _WorkspaceStub()
            def start_io(self, *a, **k):
                pass
            def feature(self, method):
                def _decor(fn):
                    return fn
                return _decor
            def command(self, name):
                def _decor(fn):
                    return fn
                return _decor
            def notify(self, method: str, params=None):
                # No-op notification for tests; real pygls will forward to client
                return None
            def start_tcp(self, host, port):
                # Minimal start hook for main(); tests won't actually open sockets
                return None

# pygls 1.x uses the separate lsprotocol package for LSP types
from lsprotocol.types import (
    CompletionItem,
    CompletionItemKind,
    CompletionList,
    Hover,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
    CodeAction,
    CodeActionKind,
    # CodeActionParams,
    WorkspaceEdit,
    TextEdit,
    InsertTextFormat,
)

ADAPTER_URL = os.environ.get("AGI_ADAPTER_URL", "http://127.0.0.1:8765/assist")
ADAPTER_TIMEOUT = float(os.environ.get("AGI_ADAPTER_TIMEOUT", "30.0"))
LOG = logging.getLogger("agi_lsp")
ADAPTER_MAX_ATTEMPTS = int(os.environ.get("AGI_ADAPTER_MAX_ATTEMPTS", "3"))
ADAPTER_BACKOFF = float(os.environ.get("AGI_ADAPTER_BACKOFF", "0.5"))
ADAPTER_STREAM_URL = ADAPTER_URL.rstrip('/') + '/stream'


class AGI_LS(LanguageServer):
    CMD = "agi_lsp"

    def __init__(self, name: str = "agi_lsp", version: str = "0.1.0"):
        super().__init__(name=name, version=version)


ls = AGI_LS()


def _get_document_safe(ls: AGI_LS, uri: str):
    """Get a document from the language server workspace, with a safe fallback
    when the workspace is not available (tests use an uninitialized server).
    """
    try:
        return ls.workspace.get_document(uri)
    except Exception:
        # Return a minimal document-like object
        class _DocStub:
            def __init__(self, u):
                self.source = ''
                self.uri = u
                self.path = None
                self.language_id = 'plaintext'

        return _DocStub(uri)


def post_with_retry(url: str, json_payload: dict, timeout: float, max_attempts: int = ADAPTER_MAX_ATTEMPTS, backoff_factor: float = ADAPTER_BACKOFF) -> httpx.Response:
    """POST to the adapter with retries, exponential backoff and jitter.

    Retries on network errors and 5xx responses. Does not retry on 4xx client errors.
    """
    attempt = 1
    last_exc: Optional[Exception] = None
    while attempt <= max_attempts:
        try:
            resp = httpx.post(url, json=json_payload, timeout=timeout)
            # Raise for HTTP errors so we can inspect status codes
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            # Retry on server errors (5xx), otherwise raise immediately for 4xx
            if status is not None and 500 <= status < 600:
                last_exc = e
            else:
                raise
        except httpx.RequestError as e:
            last_exc = e

        # If we'll retry, sleep with exponential backoff + small jitter
        if attempt < max_attempts:
            sleep_for = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, backoff_factor * 0.1)
            LOG.warning("Adapter request attempt %d/%d failed; retrying in %.2fs: %s", attempt, max_attempts, sleep_for, last_exc)
            time.sleep(sleep_for)
        attempt += 1

    # Retries exhausted
    if last_exc:
        raise last_exc
    raise RuntimeError("Adapter request failed with unknown error")


def build_assist_payload(document_text: str, position: Position):
    # Provide the whole document and the cursor offset (approx via line/char)
    # compute offset
    lines = document_text.splitlines(keepends=True)
    offset = 0
    for i in range(position.line):
        offset += len(lines[i]) if i < len(lines) else 0
    offset += position.character
    payload = {
        "file_path": None,
        "code": document_text,
        "cursor_offset": offset,
        "instruction": "Provide code-completion suggestions and a short explanation. Return a compact suggestion first, plus an optional explanation." ,
    }
    return payload


# Keep reference to latest streaming thread so tests can join it
_last_stream_thread: Optional[threading.Thread] = None
_stream_sessions: Dict[str, Dict[str, Any]] = {}


@ls.feature("textDocument/completion")
def completions(ls: AGI_LS, params):
    doc = _get_document_safe(ls, params.text_document.uri)
    pos = params.position
    payload = build_assist_payload(doc.source, pos)

    try:
        r = post_with_retry(ADAPTER_URL, payload, timeout=ADAPTER_TIMEOUT)
        data = r.json()
    except Exception as e:
        LOG.exception("Completion proxy failed after retries: %s", e)
        # Fallback single completion: no-op
        return CompletionList(is_incomplete=False, items=[CompletionItem(label="// AGI unavailable", kind=CompletionItemKind.Text)])

    suggestion = data.get("suggestion") if isinstance(data, dict) else None
    if not suggestion:
        suggestion = json.dumps(data, ensure_ascii=False) if data is not None else ""

    # Clean suggestion: remove common markdown code fences and trim
    def clean_suggestion_text(text: str) -> str:
        if not text:
            return ""
        txt = text.replace("\r\n", "\n")
        # If suggestion contains fenced code blocks, extract inner content
        if "```" in txt:
            parts = txt.split("```")
            # If fenced block exists, try to find the largest inner block
            inner_blocks = [p for i, p in enumerate(parts) if i % 2 == 1]
            if inner_blocks:
                # Prefer the first fenced block's content
                txt = inner_blocks[0]
        # Trim leading/trailing whitespace
        return txt.strip()

    cleaned = clean_suggestion_text(suggestion)

    # Provide a single snippet completion item containing the whole cleaned suggestion.
    label_preview = (cleaned.splitlines()[0] if cleaned else suggestion[:60])
    item = CompletionItem(
        label=label_preview,
        detail="AGI suggestion",
        kind=CompletionItemKind.Snippet,
        insert_text=cleaned,
        insert_text_format=InsertTextFormat.Snippet,
    )

    return CompletionList(is_incomplete=False, items=[item])


@ls.feature("textDocument/hover")
def hover(ls: AGI_LS, params):
    doc = _get_document_safe(ls, params.text_document.uri)
    pos = params.position
    payload = build_assist_payload(doc.source, pos)
    payload["instruction"] = "Provide a short explanation of the code at the cursor and suggest improvements."

    try:
        r = post_with_retry(ADAPTER_URL, payload, timeout=ADAPTER_TIMEOUT)
        data = r.json()
    except Exception as e:
        LOG.exception("Hover proxy failed after retries: %s", e)
        return None

    suggestion = data.get("suggestion") if isinstance(data, dict) else None
    if not suggestion:
        suggestion = json.dumps(data, ensure_ascii=False)

    contents = MarkupContent(kind=MarkupKind.PlainText, value=suggestion)
    return Hover(contents=contents)


@ls.feature("textDocument/codeAction")
def code_actions(ls: AGI_LS, params):
    """Request AGI-driven code actions (quick fixes / refactors) from the adapter.

    Expects the adapter to return JSON with `code_actions`: a list of
    {title, range: {start_line,start_col,end_line,end_col}, new_text} entries.
    """
    doc = _get_document_safe(ls, params.text_document.uri)

    sel_range = params.range
    # Collect selection text lines safely
    lines = doc.source.splitlines()
    start_line = max(0, sel_range.start.line)
    end_line = max(start_line, sel_range.end.line)
    sel_text = "\n".join(lines[start_line:end_line + 1]) if lines else ""

    # Normalize diagnostics into simple serializable dicts
    diagnostics = []
    for d in getattr(params.context, 'diagnostics', []) or []:
        try:
            diagnostics.append({
                'range': {
                    'start_line': d.range.start.line,
                    'start_col': d.range.start.character,
                    'end_line': d.range.end.line,
                    'end_col': d.range.end.character,
                },
                'message': getattr(d, 'message', None),
                'severity': getattr(d, 'severity', None),
                'code': getattr(d, 'code', None),
            })
        except Exception:
            continue

    payload = {
        'mode': 'code_actions',
        'file_path': getattr(doc, 'path', None),
        'uri': doc.uri,
        'language': getattr(doc, 'language_id', None),
        'selection': {
            'start_line': sel_range.start.line,
            'start_col': sel_range.start.character,
            'end_line': sel_range.end.line,
            'end_col': sel_range.end.character,
            'text': sel_text,
        },
        'diagnostics': diagnostics,
    }

    try:
        r = post_with_retry(ADAPTER_URL, payload, timeout=ADAPTER_TIMEOUT)
        data = r.json()
    except Exception as e:
        LOG.exception('AGI code_actions error: %s', e)
        return []

    suggestions = data.get('code_actions') or data.get('edits') or []
    if not isinstance(suggestions, list):
        return []

    actions = []
    for s in suggestions:
        title = s.get('title') or 'AGI suggestion'
        edit_range = s.get('range')
        new_text = s.get('new_text')

        if not edit_range or new_text is None:
            continue

        rng = Range(
            start=Position(line=edit_range['start_line'], character=edit_range['start_col']),
            end=Position(line=edit_range['end_line'], character=edit_range['end_col']),
        )

        text_edit = TextEdit(range=rng, new_text=new_text)
        ws_edit = WorkspaceEdit(changes={doc.uri: [text_edit]})

        action = CodeAction(title=title, kind=CodeActionKind.QuickFix, edit=ws_edit)
        actions.append(action)

    return actions


@ls.command("agi_lsp.apply_suggestion")
def apply_suggestion(ls: AGI_LS, *args):
    # This is a placeholder for editor-initiated apply commands; editors can call this
    LOG.info("apply_suggestion called with args=%s", args)


@ls.command("agi_lsp.cancel_stream")
def cancel_stream(ls: AGI_LS, session_id: Optional[str] = None):
    """Cancel a running streaming session by `session_id`. If `session_id` is None,
    attempt to cancel all sessions.
    """
    try:
        if session_id:
            s = _stream_sessions.get(session_id)
            if s and 'stop_event' in s:
                s['stop_event'].set()
                LOG.info("Cancelled stream session %s", session_id)
        else:
            for sid, s in list(_stream_sessions.items()):
                if 'stop_event' in s:
                    s['stop_event'].set()
            LOG.info("Cancelled all stream sessions")
    except Exception:
        LOG.exception("Error cancelling stream %s", session_id)


@ls.feature("textDocument/streamingCompletion")
def streaming_completion(ls: AGI_LS, params):
    """Initiate a streaming completion: spawn a background worker that connects
    to the adapter's stream endpoint and forwards incremental chunks to the client
    via a custom notification `agi/completionChunk`.

    The notification payload is: {"uri": <doc uri>, "chunk": <text>, "final": bool}
    """
    doc = _get_document_safe(ls, params.text_document.uri)
    pos = params.position
    payload = build_assist_payload(doc.source, pos)
    payload["stream"] = True

    # Create a session id and stop event to allow cancellation/cleanup
    session_id = str(uuid.uuid4())
    stop_event = threading.Event()
    session: Dict[str, Any] = {
        'thread': None,
        'stop_event': stop_event,
        'buffer': [],
        'last_send': 0.0,
    }
    _stream_sessions[session_id] = session

    def worker():
        try:
            with httpx.Client(timeout=ADAPTER_TIMEOUT) as client:
                # Attempt stream connection and forward chunks
                with client.stream("POST", ADAPTER_STREAM_URL, json=payload) as resp:
                    if resp.status_code != 200:
                        LOG.error("Stream upstream returned status %s", resp.status_code)
                        ls.notify("window/showMessage", {"type": 3, "message": f"AGI stream error: {resp.status_code}"})
                        return

                    started = False
                    content_type = resp.headers.get("content-type", "")

                    def flush_buffer(final: bool = False):
                        if not session['buffer'] and not final:
                            return
                        text = "".join(session['buffer']).strip()
                        session['buffer'].clear()
                        # Send assembled progress notification
                        ls.notify("agi/completionChunk", {"session_id": session_id, "uri": params.text_document.uri, "text": text, "final": final})
                        session['last_send'] = time.time()

                    if "text/event-stream" in content_type:
                        for chunk in resp.iter_bytes():
                            if stop_event.is_set():
                                LOG.info("Stream session %s cancelled by stop_event", session_id)
                                break
                            if not chunk:
                                continue
                            started = True
                            try:
                                txt = chunk.decode("utf-8", errors="ignore")
                            except Exception:
                                txt = str(chunk)
                            # Normalize and append
                            session['buffer'].append(txt)
                            # Flush periodically (every 0.5s)
                            if time.time() - session['last_send'] > 0.5:
                                flush_buffer(final=False)
                    else:
                        for line in resp.iter_lines():
                            if stop_event.is_set():
                                LOG.info("Stream session %s cancelled by stop_event", session_id)
                                break
                            if line is None:
                                continue
                            started = True
                            line = line.rstrip("\n\r")
                            if not line:
                                continue
                            payload_line = line[5:].strip() if line.startswith("data:") else line
                            session['buffer'].append(payload_line)
                            if time.time() - session['last_send'] > 0.5:
                                flush_buffer(final=False)

                    # Final flush and mark final=true (single final notification)
                    flush_buffer(final=True)
        except Exception as e:
            LOG.exception("Streaming worker error: %s", e)
            try:
                ls.notify("window/showMessage", {"type": 3, "message": f"AGI stream failed: {e}"})
            except Exception:
                LOG.debug("Failed to notify client about stream error: %s", traceback.format_exc())
        finally:
            # Clean up session
            try:
                _stream_sessions.pop(session_id, None)
            except Exception:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    session['thread'] = thread
    global _last_stream_thread
    _last_stream_thread = thread
    thread.start()
    # Return a small acknowledgement containing the session id so clients can cancel
    return {"session_id": session_id}



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run AGI LSP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2087)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    LOG.info("Starting AGI LSP; adapter=%s", ADAPTER_URL)
    ls.start_tcp(args.host, args.port)


if __name__ == "__main__":
    main()
