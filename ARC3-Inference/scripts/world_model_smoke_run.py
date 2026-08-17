#!/usr/bin/env python
"""Offline end-to-end smoke run for the duck sandbox's world-model machinery
(``save_model`` / ``run_backtest``), driven through the REAL ``HarnessSolver``
/ ``ToolAgent`` / sandbox stack against a real TAAF benchmark.

There is no GPU/OpenRouter key available here, and the offline ARC game
files are not bundled in this build, so this uses the two offline pieces
that *do* exist:

- ``taaf.game_examples.ExampleGame`` — a tiny deterministic two-level game.
- A stub OpenAI-compatible model server (stdlib ``http.server`` only) that
  always emits one ``python`` tool call. The emitted code is not scripted
  to "look right" — it inspects ``current_frame`` at sandbox runtime to
  decide UP vs DOWN, and calls the real ``save_model``/``run_backtest``
  sandbox globals exactly as an LLM would.

Run from ``ARC3-Inference/``:

    uv run --no-sync python scripts/world_model_smoke_run.py

This does NOT touch python_tool_sandbox.py, tool_agent.py, or prompts.py.
It only observes the branch as built, through real environment variables
that ``ToolAgent``/``HarnessSolver`` already read.
"""
from __future__ import annotations

import http.server
import json
import socket
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_DIR = REPO_ROOT / "runs" / "world-model-smoke"
STUB_MODEL_ID = "stub-world-model"

# ---------------------------------------------------------------------------
# The Python source handed to save_model(source). It is executed by the real
# sandbox host (python_tool_sandbox.py's _exec_world_model) exactly like a
# model-authored world model would be. No backslashes are needed anywhere in
# it (encode() reads frame.ascii[0] directly), so it survives being embedded
# once inside the outer tool-call code below without any escaping surprises.
# ---------------------------------------------------------------------------
WORLD_MODEL_SOURCE = '''
FILL_TO_CHAR = {0: "W", 1: "w", 2: "g", 4: "c", 5: "B", 6: "M", 14: "N"}
CHAR_TO_FILL = {}
for _fill_value, _char_value in FILL_TO_CHAR.items():
    CHAR_TO_FILL[_char_value] = _fill_value
LEVEL_TARGET_ACTION = {0: "UP", 1: "DOWN"}

def encode(frame):
    return frame.ascii[0]

def step(state, action):
    if state == "N":
        return "N"
    fill = CHAR_TO_FILL.get(state)
    if fill is None:
        return state
    level = fill // 4
    progress = fill % 4
    target_action = LEVEL_TARGET_ACTION.get(level, "UP")
    normalized_action = str(action or "").strip().upper()
    if normalized_action == target_action:
        progress = progress + 1
        if progress >= 3:
            level = level + 1
            progress = 0
            if level >= 2:
                return "N"
    else:
        progress = 0
    new_fill = (level * 4 + progress) % 16
    return FILL_TO_CHAR.get(new_fill, state)
'''

# The outer python-tool code, sent as the model's tool call every turn. It is
# scripted at the level of "what a diligent model would do", but it makes its
# UP/DOWN decision by reading current_frame at sandbox runtime, not by any
# out-of-band knowledge of which turn this is.
TOOL_CODE = '''
frame = current_frame
board_text = frame.ascii
first_char = board_text[0]
distinct_chars = sorted(set(board_text.replace(chr(10), "")))
print("current_frame summary: shape=" + str(frame.shape) + " step=" + str(frame.step) + " level=" + str(frame.level) + " distinct_chars=" + str(distinct_chars))

world_model_source = r"""''' + WORLD_MODEL_SOURCE + '''"""

save_result = save_model(world_model_source)
print("save_model(source) -> " + str(save_result))

backtest_report = run_backtest()
print("run_backtest() -> " + str(backtest_report))

if first_char == "N":
    print("frame already shows the WIN fill; taking no further action")
elif first_char in ("W", "w", "g"):
    action(["UP"])
elif first_char in ("c", "B", "M"):
    action(["DOWN"])
else:
    print("unrecognized fill char " + repr(first_char) + "; defaulting to UP")
    action(["UP"])
'''


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _StubModelHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    request_count = 0
    _lock = threading.Lock()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # keep stdout clean; the smoke run prints its own progress

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path.rstrip("/") == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": STUB_MODEL_ID, "object": "model", "owned_by": "stub"}],
                },
            )
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            request_payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            request_payload = {}

        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._write_json(404, {"error": "not found"})
            return

        with _StubModelHandler._lock:
            _StubModelHandler.request_count += 1
            call_index = _StubModelHandler.request_count

        model_id = str(request_payload.get("model") or STUB_MODEL_ID)
        response_payload = {
            "id": f"stub-chatcmpl-{call_index}",
            "object": "chat.completion",
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "World model: uniform 64x64 fill grid; the fill character encodes "
                            "(level, in-level progress). Plan: save encode/step, backtest, then "
                            "act toward the next progress step."
                        ),
                        "tool_calls": [
                            {
                                "id": f"call_{call_index}",
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": json.dumps({"code": TOOL_CODE}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        self._write_json(200, response_payload)


def _start_stub_server() -> tuple[http.server.ThreadingHTTPServer, int, threading.Thread]:
    port = _pick_free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _StubModelHandler)
    thread = threading.Thread(target=server.serve_forever, name="stub-model-server", daemon=True)
    thread.start()
    return server, port, thread


def main() -> int:
    server, port, thread = _start_stub_server()
    base_url = f"http://127.0.0.1:{port}/v1"
    print(f"Stub OpenAI-compatible model server listening on {base_url}")

    # These must be set BEFORE inference.agent.tool_agent (or anything that
    # imports it, like inference.framework.solver) is imported: several of
    # them are read once into module-level constants at import time.
    import os

    os.environ["LOCAL_ANALYZER_BASE_URL"] = base_url
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["LOCAL_ANALYZER_MODEL_ID"] = STUB_MODEL_ID
    os.environ["INFERENCE_ANALYZER_MODEL"] = STUB_MODEL_ID
    os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
    os.environ["OPENAI_PROVIDER"] = "vllm"
    os.environ.pop("MULTIMODAL_CONTEXT", None)
    # Avoid a stray key turning this into an authenticated call anywhere.
    os.environ.pop("LOCAL_ANALYZER_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENROUTER_API_KEY", None)

    import asyncio

    import taaf.benchmark
    import taaf.deploy_inline
    import taaf.game_examples

    from inference.framework.solver import HarnessSolver

    if JOB_DIR.exists():
        print(f"Removing stale job_dir from a previous run: {JOB_DIR}")
        import shutil

        shutil.rmtree(JOB_DIR)

    solver = HarnessSolver(
        label="world-model-smoke",
        model=STUB_MODEL_ID,
        analyzer_timeout=30.0,
        max_actions_per_game=20,
        max_runtime_s_per_game=60.0,
        concurrency=1,
        save_request_logs=True,
    )
    benchmark = taaf.benchmark.Benchmark(
        label="world-model-smoke",
        games=[taaf.game_examples.ExampleGame()],
        solver=solver,
        n_passes=1,
        job_dir=JOB_DIR,
    )
    target = taaf.deploy_inline.InlineTarget(max_runtime_s=120.0)

    print(f"Run directory: {JOB_DIR}")
    print("Deploying benchmark inline...")
    handle = asyncio.run(benchmark.deploy(target))

    print(f"\nStub model server handled {_StubModelHandler.request_count} chat-completion request(s).")

    ran_benchmark = handle.benchmark
    for run in ran_benchmark.game_runs:
        print(
            "Final game state: "
            f"game_id={run.game_id} state={run.state} "
            f"levels_completed={run.levels_completed}/{run.number_of_levels} "
            f"final_score={run.final_score} solver_note={run.solver_note!r}"
        )

    server.shutdown()
    server.server_close()
    thread.join(timeout=5)

    artifacts_dir = JOB_DIR / "artifacts"
    if artifacts_dir.is_dir():
        print(f"\nArtifacts under {artifacts_dir}:")
        for path in sorted(artifacts_dir.iterdir()):
            print(f"  {path.name} ({path.stat().st_size} bytes)")
    else:
        print(f"\nNo artifacts directory found at {artifacts_dir}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
