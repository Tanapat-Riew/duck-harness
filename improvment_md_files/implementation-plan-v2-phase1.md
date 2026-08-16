# Duck Harness Phase 1 — Induced World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the duck induce an executable world model, persist it across tool calls within a game run, and certify it against recorded transitions for zero environment actions.

**Architecture:** The sandbox stays sealed — no new builtins, no filesystem. The model writes `encode`/`step` as a source string and hands it to a host-mediated `save_model(source)` bridge; the host keeps that source in memory on the `ToolAgent` instance and re-`exec`s it into the sandbox namespace at the start of every call. `run_backtest()` replays the already-present `transitions` list through those functions inside the sandbox, with no host round-trip. Alongside it, the harness starts computing two things the model cannot: a per-action board-change report and a per-level no-effect action memory.

**Tech Stack:** Python 3.12, uv, pytest 9.0.2, ruff 0.15.6. No new runtime dependencies.

**Scope:** This plan covers Phase 1 and 1b of `improvment-v2.md` only. Phase 2 (`run_bfs`), Phase 3 (`commit_actions`), and Phase 4 (representation-revision prompting) get their own plans after the go/no-go measurement in §"Closing out" below. That gate exists because if the 27B cannot certify a model on at least 3 of 25 games, Phases 2–4 have no foundation.

## Global Constraints

- All commands run from `ARC3-Inference/`.
- Full check: `make prepare-ci` (ruff + pytest). Single test: `uv run --locked --extra dev pytest tests/<file>::<test> -v`.
- `tests/` does not exist in this checkout. `pyproject.toml` already sets `testpaths = ["tests"]`; Task 1 creates the directory.
- **Do not add entries to `SAFE_BUILTINS` or `SAFE_MODULES`** in `python_tool_sandbox.py`. The sealed sandbox is a design invariant, not an obstacle.
- **Every new `ToolAgent` attribute must be read with `getattr(self, name, default)`.** The Kaggle bundle unpickles a `__dict__` written before these attributes existed; a bare `self._new_attr` read raises `AttributeError` on an old pickle.
- **No cross-pass persistence.** All new state lives on the `ToolAgent` instance and is cleared in `_ensure_session`. Never write it to a path that outlives the game run.
- Commit messages: plain imperative sentence, matching existing history (`Refactor README for clarity and formatting`). If an agent is committing, append the `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer.
- Sandbox source lives inside the `_SANDBOX_BOOTSTRAP` string literal in `python_tool_sandbox.py`. Edits there are edits to a string — indentation is four spaces deeper than a normal module.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `inference/utils/board_change.py` | **new** — pure grid diff and border classification. No imports from `inference.agent`. |
| `inference/agent/tool_agent.py` | host side: change report on action results, world-model source store, backtest result store, no-effect memory, prompt lines. |
| `inference/agent/python_tool_sandbox.py` | sandbox side: `save_model`, `run_backtest`, world-model reload, extra response keys. |
| `inference/agent/prompts.py` | **new addendum** `WORLD_MODEL_ADDENDUM`. |
| `tests/test_board_change.py` | **new** — Task 1. |
| `tests/test_tool_agent_change_report.py` | **new** — Task 2. |
| `tests/test_sandbox_world_model.py` | **new** — Tasks 3 and 4, integration through a real sandbox subprocess. |
| `tests/test_tool_agent_world_model_prompt.py` | **new** — Task 5. |
| `tests/test_tool_agent_effect_memory.py` | **new** — Task 6. |

---

### Task 1: Board-change classifier

Pure function, no agent coupling. Everything downstream depends on it.

**Files:**
- Create: `inference/utils/board_change.py`
- Create: `tests/test_board_change.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `classify_board_change(before, after, *, border_width=2) -> BoardChangeReport`, `diff_cells(before, after) -> tuple[tuple[int, int], ...]`, and `BoardChangeReport.as_dict() -> dict` with keys `changed_count: int`, `changed_cells: list[list[int]]` (max 20), `bounding_box: list[int] | None` as `[r0, c0, r1, c1]`, `border_only: bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_board_change.py`:

```python
from inference.utils.board_change import classify_board_change, diff_cells


def _blank(rows, cols):
    return tuple(tuple(0 for _ in range(cols)) for _ in range(rows))


def _with_changes(grid, cells, value=3):
    rows = [list(row) for row in grid]
    for row, col in cells:
        rows[row][col] = value
    return tuple(tuple(row) for row in rows)


def test_identical_grids_report_no_change():
    grid = _blank(8, 8)
    report = classify_board_change(grid, grid)
    assert report.changed_count == 0
    assert report.changed_cells == ()
    assert report.bounding_box is None
    assert report.border_only is False


def test_interior_change_is_not_border_only():
    before = _blank(10, 10)
    after = _with_changes(before, [(5, 5)])
    report = classify_board_change(before, after)
    assert report.changed_count == 1
    assert report.bounding_box == (5, 5, 5, 5)
    assert report.border_only is False


def test_top_edge_strip_is_border_only():
    before = _blank(10, 10)
    after = _with_changes(before, [(0, col) for col in range(10)])
    report = classify_board_change(before, after)
    assert report.changed_count == 10
    assert report.border_only is True


def test_mixed_edge_and_interior_is_not_border_only():
    before = _blank(10, 10)
    after = _with_changes(before, [(0, 0), (5, 5)])
    report = classify_board_change(before, after)
    assert report.changed_count == 2
    assert report.border_only is False


def test_ragged_and_mismatched_shapes_do_not_crash():
    before = ((0, 0, 0), (0, 0))
    after = ((0, 1, 0),)
    report = classify_board_change(before, after)
    assert report.changed_count == 3
    assert diff_cells(before, after) == ((0, 1), (1, 0), (1, 1))


def test_as_dict_truncates_the_cell_list():
    before = _blank(30, 30)
    after = _with_changes(before, [(row, 10) for row in range(25)])
    payload = classify_board_change(before, after).as_dict()
    assert payload["changed_count"] == 25
    assert len(payload["changed_cells"]) == 20
    assert payload["changed_cells"][0] == [0, 10]
    assert payload["bounding_box"] == [0, 10, 24, 10]
    assert payload["border_only"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --locked --extra dev pytest tests/test_board_change.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'inference.utils.board_change'`

- [ ] **Step 3: Write the implementation**

Create `inference/utils/board_change.py`:

```python
"""Classify what changed between two board grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

Grid = Sequence[Sequence[int]]

MAX_REPORTED_CELLS = 20


@dataclass(frozen=True)
class BoardChangeReport:
    """Summary of the cells that differ between two grids."""

    changed_count: int
    changed_cells: tuple[tuple[int, int], ...]
    bounding_box: tuple[int, int, int, int] | None
    border_only: bool

    def as_dict(self) -> dict[str, Any]:
        """Render the report as a compact JSON-safe dict for tool results."""
        return {
            "changed_count": self.changed_count,
            "changed_cells": [list(cell) for cell in self.changed_cells[:MAX_REPORTED_CELLS]],
            "bounding_box": list(self.bounding_box) if self.bounding_box is not None else None,
            "border_only": self.border_only,
        }


def _grid_shape(grid: Grid) -> tuple[int, int]:
    """Return (rows, cols) tolerating ragged rows."""
    return len(grid), max((len(row) for row in grid), default=0)


def diff_cells(before: Grid, after: Grid) -> tuple[tuple[int, int], ...]:
    """Return every (row, col) whose value differs, treating missing cells as None."""
    changed: list[tuple[int, int]] = []
    for row_index in range(max(len(before), len(after))):
        before_row = before[row_index] if row_index < len(before) else ()
        after_row = after[row_index] if row_index < len(after) else ()
        for col_index in range(max(len(before_row), len(after_row))):
            before_value = before_row[col_index] if col_index < len(before_row) else None
            after_value = after_row[col_index] if col_index < len(after_row) else None
            if before_value != after_value:
                changed.append((row_index, col_index))
    return tuple(changed)


def classify_board_change(before: Grid, after: Grid, *, border_width: int = 2) -> BoardChangeReport:
    """Diff two grids and report whether the change is confined to a border strip.

    ``border_only`` is the signal that an action moved only a HUD or timer bar
    rather than gameplay state. It is False when nothing changed at all.
    """
    changed = diff_cells(before, after)
    if not changed:
        return BoardChangeReport(0, (), None, False)
    before_rows, before_cols = _grid_shape(before)
    after_rows, after_cols = _grid_shape(after)
    rows = max(before_rows, after_rows)
    cols = max(before_cols, after_cols)
    row_values = [cell[0] for cell in changed]
    col_values = [cell[1] for cell in changed]
    bounding_box = (min(row_values), min(col_values), max(row_values), max(col_values))
    border_only = all(
        row < border_width
        or row >= rows - border_width
        or col < border_width
        or col >= cols - border_width
        for row, col in changed
    )
    return BoardChangeReport(len(changed), changed, bounding_box, border_only)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --locked --extra dev pytest tests/test_board_change.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Commit**

```bash
git add inference/utils/board_change.py tests/test_board_change.py && git commit -m "Add board-change classifier with border-strip detection"
```

---

### Task 2: Attach the change report to action results

**Files:**
- Modify: `inference/agent/tool_agent.py` (imports near line 51; new helper after `_display_action_number`; `_handle_action` around lines 1810–1826)
- Create: `tests/test_tool_agent_change_report.py`

**Interfaces:**
- Consumes: `classify_board_change` from Task 1.
- Produces: `_frame_change_report(before: Frame | None, after: Frame | None) -> dict[str, Any] | None`, and a `change_report` key on every executed action result dict.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tool_agent_change_report.py`:

```python
from inference.agent.runtime_state import Frame
from inference.agent.tool_agent import _frame_change_report


def _frame(cells, *, step=0, level=1, rows=10, cols=10):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for row, col in cells:
        grid[row][col] = 5
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


def test_report_flags_a_bottom_edge_strip_as_border_only():
    before = _frame([])
    after = _frame([(9, col) for col in range(10)], step=1)
    report = _frame_change_report(before, after)
    assert report["border_only"] is True
    assert report["changed_count"] == 10


def test_report_flags_an_interior_move_as_gameplay():
    before = _frame([(4, 4)])
    after = _frame([(4, 5)], step=1)
    report = _frame_change_report(before, after)
    assert report["border_only"] is False
    assert report["changed_count"] == 2


def test_report_is_none_when_a_frame_is_missing():
    assert _frame_change_report(None, _frame([])) is None
    assert _frame_change_report(_frame([]), None) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --locked --extra dev pytest tests/test_tool_agent_change_report.py -v`
Expected: FAIL — `ImportError: cannot import name '_frame_change_report'`

- [ ] **Step 3: Add the helper**

In `inference/agent/tool_agent.py`, add to the import block near line 51:

```python
from inference.utils.board_change import classify_board_change
```

Then add this function immediately after `_display_action_number` (around line 285):

```python
def _frame_change_report(before: Frame | None, after: Frame | None) -> dict[str, Any] | None:
    """Summarize the board change between two frames, or None if either is missing."""
    if before is None or after is None:
        return None
    return classify_board_change(before.grid, after.grid).as_dict()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --locked --extra dev pytest tests/test_tool_agent_change_report.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Wire the report into executed actions**

In `_handle_action` inside `_run_python_tool`, replace this block:

```python
            raw_payload = self._step_env_callback({"actions": normalized_actions})
            if not isinstance(raw_payload, dict):
                raise RuntimeError("action(actions) did not return a JSON-like payload.")
            compact_payload = self._compact_action_result(raw_payload)
```

with:

```python
            before_frame, _ = load_runtime_state(state_path)
            raw_payload = self._step_env_callback({"actions": normalized_actions})
            if not isinstance(raw_payload, dict):
                raise RuntimeError("action(actions) did not return a JSON-like payload.")
            compact_payload = self._compact_action_result(raw_payload)
            after_frame, _ = load_runtime_state(state_path)
            change_report = _frame_change_report(before_frame, after_frame)
            if change_report is not None:
                compact_payload["change_report"] = change_report
```

Note: for a batched `action(['UP', 'UP', 'UP'])` this reports the net change across the whole batch, which is the correct granularity — one `action()` call is one observation.

- [ ] **Step 6: Verify nothing regressed**

Run: `make prepare-ci`
Expected: ruff clean, all tests pass

- [ ] **Step 7: Commit**

```bash
git add inference/agent/tool_agent.py tests/test_tool_agent_change_report.py && git commit -m "Report board-change geometry on every executed action"
```

---

### Task 3: Persist an induced world model across tool calls

**Files:**
- Modify: `inference/agent/python_tool_sandbox.py` (inside `_SANDBOX_BOOTSTRAP`: `main()` around lines 320–392; host return around line 560)
- Modify: `inference/agent/tool_agent.py` (`__init__` ~line 1158, `_ensure_session` ~line 1189, `_serialized_runtime_state` ~line 1758, after `run_sandboxed_python` ~line 1833)
- Create: `tests/test_sandbox_world_model.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: sandbox global `save_model(source: str) -> dict` with keys `ok: bool`, `error: str` (on failure), `defines: list[str]` (on success); sandbox globals `encode`, `step`, and optionally `is_goal` after a successful save or reload; `run_sandboxed_python(...)` return dict gains `world_model_source: str`; `ToolAgent._world_model_source: str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox_world_model.py`:

```python
from inference.agent.python_tool_sandbox import run_sandboxed_python

CORRECT_MODEL = '''
def encode(frame):
    return frame.ascii.index("x")

def step(state, action):
    if action == "RIGHT":
        return state + 1
    return state
'''


def _frame_payload(step_index, ascii_text):
    return {
        "ascii": ascii_text,
        "step": step_index,
        "level": 1,
        "shape": [1, 3],
        "grid": [[0, 0, 0]],
    }


def _initial_state(world_model_source=""):
    return {
        "current_frame": _frame_payload(2, "..x"),
        "history": [
            {"action": "", "frame": _frame_payload(0, "x..")},
            {"action": "RIGHT", "frame": _frame_payload(1, ".x.")},
            {"action": "RIGHT", "frame": _frame_payload(2, "..x")},
        ],
        "valid_actions": ["RIGHT"],
        "last_action_result": {},
        "world_model_source": world_model_source,
    }


def _no_actions(actions):
    raise AssertionError("this test must not step the environment")


def _run(code, world_model_source=""):
    return run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state=_initial_state(world_model_source),
        action_handler=_no_actions,
    )


def test_save_model_accepts_a_valid_model_and_returns_the_source():
    outcome = _run(f'result = save_model("""{CORRECT_MODEL}""")')
    assert outcome["result"]["ok"] is True
    assert outcome["result"]["defines"] == ["encode", "step"]
    assert "def step(state, action):" in outcome["world_model_source"]


def test_save_model_rejects_a_model_without_step():
    outcome = _run('result = save_model("def encode(frame):\\n    return 1\\n")')
    assert outcome["result"]["ok"] is False
    assert "step" in outcome["result"]["error"]
    assert outcome["world_model_source"] == ""


def test_save_model_rejects_a_syntax_error():
    outcome = _run('result = save_model("def encode(frame:\\n")')
    assert outcome["result"]["ok"] is False
    assert outcome["world_model_source"] == ""


def test_a_saved_model_is_reloaded_into_the_next_call():
    outcome = _run("result = encode(current_frame)", world_model_source=CORRECT_MODEL)
    assert outcome["result"] == 2
    assert outcome["error"] == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --locked --extra dev pytest tests/test_sandbox_world_model.py -v`
Expected: FAIL — `NameError: name 'save_model' is not defined`

- [ ] **Step 3: Add `save_model` and the reload to the sandbox bootstrap**

In `inference/agent/python_tool_sandbox.py`, inside `_SANDBOX_BOOTSTRAP`, in `main()` — insert immediately before the existing `runtime_globals["action"] = action` line:

```python
        saved_world_model = {"source": ""}

        def save_model(source):
            text = str(source or "")
            namespace = {"__builtins__": runtime_globals["__builtins__"]}
            try:
                exec(compile(text, "<world_model>", "exec"), namespace, namespace)
            except Exception as exc:
                return {"ok": False, "error": _sanitize_exception(exc)}
            missing = [name for name in ("encode", "step") if not callable(namespace.get(name))]
            if missing:
                return {
                    "ok": False,
                    "error": "world model must define callable " + ", ".join(missing),
                }
            saved_world_model["source"] = text
            for name, value in namespace.items():
                if name != "__builtins__":
                    runtime_globals[name] = value
            return {
                "ok": True,
                "defines": [
                    name
                    for name in ("encode", "step", "is_goal")
                    if callable(namespace.get(name))
                ],
            }

        runtime_globals["save_model"] = save_model
```

Then replace the existing line `_refresh_state(initial.get("state") or {})` with:

```python
        _refresh_state(initial.get("state") or {})

        stored_source = str((initial.get("state") or {}).get("world_model_source") or "")
        runtime_globals["world_model_error"] = None
        if stored_source:
            try:
                exec(compile(stored_source, "<world_model>", "exec"), runtime_globals, runtime_globals)
                saved_world_model["source"] = stored_source
            except Exception as exc:
                runtime_globals["world_model_error"] = _sanitize_exception(exc)
```

- [ ] **Step 4: Return the saved source to the host**

Still inside `main()`, add `"world_model_source"` to both outgoing payloads. Replace the `_send({...})` call in the `try` branch with:

```python
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "world_model_source": saved_world_model["source"],
                }
            )
```

and the one in the `except` branch with:

```python
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    "world_model_source": saved_world_model["source"],
                }
            )
```

In the host function `run_sandboxed_python`, in the `if msg_type in {"final", "error"}:` branch, replace the return with:

```python
            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "world_model_source": str(message.get("world_model_source", "") or ""),
                }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --locked --extra dev pytest tests/test_sandbox_world_model.py -v`
Expected: PASS — 4 passed

- [ ] **Step 6: Store the source on the agent**

In `inference/agent/tool_agent.py`, add to the end of `__init__`:

```python
        self._world_model_source: str = ""
```

Add the same line to `_ensure_session`, inside the `if self._session_runtime_dir != runtime_dir:` block, alongside `self._summarized_knowledge = _empty_world_model()`:

```python
            self._world_model_source = ""
```

In `_serialized_runtime_state` inside `_run_python_tool`, add one key to the returned dict:

```python
                "world_model_source": getattr(self, "_world_model_source", ""),
```

Immediately after the `sandbox_result = run_sandboxed_python(...)` call, add:

```python
        saved_source = sandbox_result.get("world_model_source")
        if isinstance(saved_source, str) and saved_source.strip():
            self._world_model_source = saved_source
```

- [ ] **Step 7: Verify nothing regressed**

Run: `make prepare-ci`
Expected: ruff clean, all tests pass

- [ ] **Step 8: Commit**

```bash
git add inference/agent/python_tool_sandbox.py inference/agent/tool_agent.py tests/test_sandbox_world_model.py && git commit -m "Persist an induced world model across sandbox calls"
```

---

### Task 4: `run_backtest()`

**Files:**
- Modify: `inference/agent/python_tool_sandbox.py` (inside `_SANDBOX_BOOTSTRAP`, next to `save_model`; both `_send` payloads; host return)
- Modify: `inference/agent/tool_agent.py` (after `run_sandboxed_python`)
- Modify: `tests/test_sandbox_world_model.py`

**Interfaces:**
- Consumes: `save_model` and the reload from Task 3.
- Produces: sandbox global `run_backtest(max_transitions: int | None = None) -> dict` with keys `ok: bool`, `total: int`, `exact: int`, `certified: bool`, `first_mismatch: dict | None` (keys `index`, `action`, and either `predicted`/`observed` or `error`); `run_sandboxed_python(...)` return dict gains `last_backtest: dict | None`; `ToolAgent._last_backtest: dict | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sandbox_world_model.py`:

```python
WRONG_MODEL = '''
def encode(frame):
    return frame.ascii.index("x")

def step(state, action):
    return state
'''

EXPLODING_MODEL = '''
def encode(frame):
    return frame.ascii.index("x")

def step(state, action):
    raise ValueError("boom")
'''


def test_backtest_certifies_a_correct_model():
    outcome = _run("result = run_backtest()", world_model_source=CORRECT_MODEL)
    assert outcome["result"] == {
        "ok": True,
        "total": 2,
        "exact": 2,
        "certified": True,
        "first_mismatch": None,
    }
    assert outcome["last_backtest"]["certified"] is True


def test_backtest_reports_the_first_mismatch():
    outcome = _run("result = run_backtest()", world_model_source=WRONG_MODEL)
    report = outcome["result"]
    assert report["total"] == 2
    assert report["exact"] == 0
    assert report["certified"] is False
    assert report["first_mismatch"]["index"] == 0
    assert report["first_mismatch"]["action"] == "RIGHT"
    assert report["first_mismatch"]["predicted"] == "0"
    assert report["first_mismatch"]["observed"] == "1"


def test_backtest_captures_an_exception_as_a_mismatch():
    outcome = _run("result = run_backtest()", world_model_source=EXPLODING_MODEL)
    report = outcome["result"]
    assert report["certified"] is False
    assert "boom" in report["first_mismatch"]["error"]


def test_backtest_without_a_model_reports_not_ok():
    outcome = _run("result = run_backtest()")
    assert outcome["result"]["ok"] is False
    assert "save_model" in outcome["result"]["error"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --locked --extra dev pytest tests/test_sandbox_world_model.py -v`
Expected: FAIL — `NameError: name 'run_backtest' is not defined`

- [ ] **Step 3: Implement `run_backtest` in the sandbox bootstrap**

In `_SANDBOX_BOOTSTRAP`, immediately after `runtime_globals["save_model"] = save_model`, add:

```python
        last_backtest = {}

        def _truncate_repr(value, limit=400):
            text = repr(value)
            if len(text) <= limit:
                return text
            return text[:limit] + "... [" + str(len(text) - limit) + " chars omitted]"

        def run_backtest(max_transitions=None):
            encode_fn = runtime_globals.get("encode")
            step_fn = runtime_globals.get("step")
            if not callable(encode_fn) or not callable(step_fn):
                return {
                    "ok": False,
                    "error": "No world model loaded. Call save_model(source) with encode and step first.",
                }
            transitions = list(runtime_globals.get("transitions") or [])
            if max_transitions is not None:
                transitions = transitions[-int(max_transitions):]
            total = 0
            exact = 0
            first_mismatch = None
            for index, transition in enumerate(transitions):
                if transition.before_frame is None or transition.after_frame is None:
                    continue
                total += 1
                try:
                    predicted = step_fn(encode_fn(transition.before_frame), transition.action)
                    observed = encode_fn(transition.after_frame)
                except Exception as exc:
                    if first_mismatch is None:
                        first_mismatch = {
                            "index": index,
                            "action": transition.action,
                            "error": _sanitize_exception(exc),
                        }
                    continue
                if predicted == observed:
                    exact += 1
                elif first_mismatch is None:
                    first_mismatch = {
                        "index": index,
                        "action": transition.action,
                        "predicted": _truncate_repr(predicted),
                        "observed": _truncate_repr(observed),
                    }
            report = {
                "ok": True,
                "total": total,
                "exact": exact,
                "certified": total > 0 and exact == total,
                "first_mismatch": first_mismatch,
            }
            last_backtest.clear()
            last_backtest.update(report)
            return report

        runtime_globals["run_backtest"] = run_backtest
```

- [ ] **Step 4: Return the backtest to the host**

Add `"last_backtest": _json_safe(last_backtest) if last_backtest else None,` to both `_send` payloads in `main()` (the `final` and `error` branches), and add to the host's `{"final", "error"}` return dict:

```python
                    "last_backtest": message.get("last_backtest"),
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --locked --extra dev pytest tests/test_sandbox_world_model.py -v`
Expected: PASS — 8 passed

- [ ] **Step 6: Store the backtest on the agent**

In `tool_agent.py`, add to the end of `__init__` and to the reset block in `_ensure_session`:

```python
        self._last_backtest: dict[str, Any] | None = None
```

(in `_ensure_session`, indented one level further and written as `self._last_backtest = None`)

Immediately after the `saved_source` block added in Task 3, add:

```python
        backtest_report = sandbox_result.get("last_backtest")
        if isinstance(backtest_report, dict) and backtest_report.get("ok"):
            self._last_backtest = dict(backtest_report)
```

- [ ] **Step 7: Verify nothing regressed**

Run: `make prepare-ci`
Expected: ruff clean, all tests pass

- [ ] **Step 8: Commit**

```bash
git add inference/agent/python_tool_sandbox.py inference/agent/tool_agent.py tests/test_sandbox_world_model.py && git commit -m "Add run_backtest to replay recorded transitions through the induced model"
```

---

### Task 5: Tell the model the tools exist

Without this the previous three tasks are unreachable — the model has no way to learn `save_model` is available.

**Files:**
- Modify: `inference/agent/prompts.py` (new addendum after `PYTHON_ADDENDUM`)
- Modify: `inference/agent/tool_agent.py` (`_build_system_prompt` ~line 436, `_build_user_prompt` ~line 1481, `_update_summarized_knowledge_from_step_summary` ~line 1349, `analyze` transcript ~line 2077)
- Create: `tests/test_tool_agent_world_model_prompt.py`

**Interfaces:**
- Consumes: `ToolAgent._world_model_source` (Task 3), `ToolAgent._last_backtest` (Task 4).
- Produces: `ToolAgent._world_model_status_lines() -> list[str]`; `WORLD_MODEL_ADDENDUM` in `prompts.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tool_agent_world_model_prompt.py`:

```python
from inference.agent.tool_agent import ToolAgent


def _agent():
    return ToolAgent.__new__(ToolAgent)


def test_status_reports_no_model_when_none_saved():
    agent = _agent()
    agent._world_model_source = ""
    agent._last_backtest = None
    lines = agent._world_model_status_lines()
    assert any("none saved yet" in line for line in lines)
    assert any("save_model" in line for line in lines)


def test_status_reports_an_uncertified_model():
    agent = _agent()
    agent._world_model_source = "def encode(f):\n    return 0\n"
    agent._last_backtest = {
        "ok": True,
        "total": 7,
        "exact": 5,
        "certified": False,
        "first_mismatch": {"index": 5, "action": "LEFT"},
    }
    lines = agent._world_model_status_lines()
    assert any("5/7" in line for line in lines)
    assert any("not certified" in line for line in lines)
    assert any("transition 5" in line and "LEFT" in line for line in lines)


def test_status_reports_a_certified_model():
    agent = _agent()
    agent._world_model_source = "def encode(f):\n    return 0\n"
    agent._last_backtest = {
        "ok": True,
        "total": 7,
        "exact": 7,
        "certified": True,
        "first_mismatch": None,
    }
    lines = agent._world_model_status_lines()
    assert any("7/7" in line for line in lines)
    assert any("certified" in line for line in lines)


def test_status_survives_an_agent_pickled_before_these_attributes_existed():
    agent = _agent()
    lines = agent._world_model_status_lines()
    assert any("none saved yet" in line for line in lines)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --locked --extra dev pytest tests/test_tool_agent_world_model_prompt.py -v`
Expected: FAIL — `AttributeError: 'ToolAgent' object has no attribute '_world_model_status_lines'`

- [ ] **Step 3: Implement the status lines**

In `tool_agent.py`, add this method immediately after `_summarized_knowledge_lines`:

```python
    def _world_model_status_lines(self) -> list[str]:
        """Render induced-model status for the prompt (safe on pre-upgrade pickles)."""
        source = getattr(self, "_world_model_source", "") or ""
        if not source.strip():
            return [
                "Induced world model: none saved yet. Write `encode(frame)` and "
                "`step(state, action)` as a source string and pass it to `save_model(source)`."
            ]
        lines = [
            f"Induced world model: saved, {len(source.splitlines())} lines, "
            "reloaded into every `python` call automatically."
        ]
        backtest = getattr(self, "_last_backtest", None)
        if isinstance(backtest, dict) and backtest.get("total"):
            status = "certified" if backtest.get("certified") else "not certified"
            lines.append(
                f"Last backtest: {backtest.get('exact', 0)}/{backtest.get('total', 0)} "
                f"transitions reproduced exactly ({status})."
            )
            mismatch = backtest.get("first_mismatch")
            if isinstance(mismatch, dict):
                lines.append(
                    f"First mismatch at transition {mismatch.get('index')} "
                    f"on action {mismatch.get('action')}."
                )
        else:
            lines.append("Last backtest: never run. Call `run_backtest()` before trusting this model.")
        return lines
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --locked --extra dev pytest tests/test_tool_agent_world_model_prompt.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Add the system-prompt addendum**

In `inference/agent/prompts.py`, add after `PYTHON_ADDENDUM`:

```python
WORLD_MODEL_ADDENDUM = (
    "\n\nInduced world model:\n"
    "- Every environment action costs score. Testing a theory against recorded transitions costs nothing. Prefer the free test.\n"
    "- Write two functions as a source string and pass it to `save_model(source)`:\n"
    "  `encode(frame)` maps a frame to a compact hashable state, and `step(state, action)` returns the next state.\n"
    "- `encode` decides what counts as state. Leave timers, progress bars, and other HUD strips OUT of it, or they will make exact reproduction impossible.\n"
    "- A saved model is re-loaded into `encode`/`step` at the start of every later `python` call in this game. It is not shared across passes.\n"
    "- `run_backtest()` replays every recorded transition through your model and returns `total`, `exact`, `certified`, and the first mismatch. Rewrite and re-check as often as you like; it never touches the environment.\n"
    "- Read `change_report` on action results: `border_only` true means the last action moved only a HUD strip, so `encode` should ignore that region.\n"
    "- If `step` keeps failing after several rewrites, the problem is usually `encode`, not `step`. Question what the objects are and whether hidden state exists that the grid never shows.\n"
    "- If you cannot certify a model after several honest attempts, stop trying and play reactively for this level. A partial model is still useful for ruling actions out.\n"
)
```

In `tool_agent.py`, import it alongside the other addendums and add it to `_build_system_prompt` after `PYTHON_ADDENDUM`:

```python
    prompt += WORLD_MODEL_ADDENDUM
```

- [ ] **Step 6: Add the status lines to the per-turn prompt**

In `_build_user_prompt`, replace:

```python
        lines.extend(self._summarized_knowledge_lines())
        lines.append("end of world model. ")
```

with:

```python
        lines.extend(self._summarized_knowledge_lines())
        lines.extend(self._world_model_status_lines())
        lines.append("end of world model. ")
```

- [ ] **Step 7: Keep the induced model across level transitions**

In `_update_summarized_knowledge_from_step_summary`, the prose fields are cleared on a level boundary. The induced model must survive — mechanics transfer between levels even when layouts do not. Add this comment above the `for key in (...)` loop so a later reader does not "fix" the omission:

```python
            # The induced world model (`_world_model_source`) deliberately survives
            # a level transition: mechanics carry across levels even when layouts
            # do not. Only the prose fields below are level-scoped.
```

- [ ] **Step 8: Log the backtest in the transcript**

In `analyze`, immediately after the `append_transcript("USER PROMPT", user_prompt)` call, add:

```python
        backtest_status = getattr(self, "_last_backtest", None)
        if isinstance(backtest_status, dict):
            append_transcript(
                "WORLD MODEL",
                f"backtest: {backtest_status.get('exact', 0)}/{backtest_status.get('total', 0)} "
                f"certified={bool(backtest_status.get('certified'))}",
            )
```

This is the line the go/no-go measurement greps for.

- [ ] **Step 9: Verify nothing regressed**

Run: `make prepare-ci`
Expected: ruff clean, all tests pass

- [ ] **Step 10: Commit**

```bash
git add inference/agent/prompts.py inference/agent/tool_agent.py tests/test_tool_agent_world_model_prompt.py && git commit -m "Surface the induced world model and backtest status in prompts"
```

---

### Task 6: No-effect action memory

The harness owns this because the model cannot — it needs state across turns.

**Files:**
- Modify: `inference/agent/tool_agent.py` (`__init__`, `_ensure_session`, `_handle_action`, `_serialized_runtime_state`, `_world_model_status_lines`)
- Create: `tests/test_tool_agent_effect_memory.py`

**Interfaces:**
- Consumes: the `change_report` key from Task 2, `_world_model_status_lines` from Task 5.
- Produces: `ToolAgent._record_action_effect(action_display: str, change_report: dict[str, Any] | None, *, level_changed: bool) -> None` — note `level_changed` is keyword-only; `ToolAgent._no_effect_actions: dict[str, int]`; sandbox global `no_effect_actions: list[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tool_agent_effect_memory.py`:

```python
from inference.agent.tool_agent import ToolAgent


def _agent():
    agent = ToolAgent.__new__(ToolAgent)
    agent._no_effect_actions = {}
    return agent


def _report(changed_count, border_only=False):
    return {
        "changed_count": changed_count,
        "changed_cells": [],
        "bounding_box": None,
        "border_only": border_only,
    }


def test_an_action_with_no_change_is_remembered():
    agent = _agent()
    agent._record_action_effect("LEFT", _report(0), level_changed=False)
    assert agent._no_effect_actions == {"LEFT": 1}


def test_repeated_no_effect_increments_the_count():
    agent = _agent()
    agent._record_action_effect("LEFT", _report(0), level_changed=False)
    agent._record_action_effect("LEFT", _report(0), level_changed=False)
    assert agent._no_effect_actions == {"LEFT": 2}


def test_an_action_that_changes_the_board_is_forgotten():
    agent = _agent()
    agent._record_action_effect("LEFT", _report(0), level_changed=False)
    agent._record_action_effect("LEFT", _report(4), level_changed=False)
    assert agent._no_effect_actions == {}


def test_a_hud_only_change_still_counts_as_no_effect():
    agent = _agent()
    agent._record_action_effect("SPACE", _report(8, border_only=True), level_changed=False)
    assert agent._no_effect_actions == {"SPACE": 1}


def test_a_level_change_clears_the_memory():
    agent = _agent()
    agent._record_action_effect("LEFT", _report(0), level_changed=False)
    agent._record_action_effect("UP", _report(3), level_changed=True)
    assert agent._no_effect_actions == {}


def test_a_missing_report_is_ignored():
    agent = _agent()
    agent._record_action_effect("LEFT", None, level_changed=False)
    assert agent._no_effect_actions == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --locked --extra dev pytest tests/test_tool_agent_effect_memory.py -v`
Expected: FAIL — `AttributeError: 'ToolAgent' object has no attribute '_record_action_effect'`

- [ ] **Step 3: Implement the memory**

In `tool_agent.py`, add to the end of `__init__`:

```python
        self._no_effect_actions: dict[str, int] = {}
```

Add to the reset block in `_ensure_session`:

```python
            self._no_effect_actions = {}
```

Add this method immediately after `_world_model_status_lines`:

```python
    def _record_action_effect(
        self,
        action_display: str,
        change_report: dict[str, Any] | None,
        *,
        level_changed: bool,
    ) -> None:
        """Track actions that moved nothing, so the model stops re-testing them.

        A HUD-only change counts as no effect: a moving timer bar is not evidence
        that the action did anything. The memory is level-scoped because an action
        that is dead on one level often works on the next.
        """
        memory = getattr(self, "_no_effect_actions", None)
        if memory is None:
            memory = {}
            self._no_effect_actions = memory
        if level_changed:
            memory.clear()
            return
        name = str(action_display or "").strip()
        if not name or not isinstance(change_report, dict):
            return
        moved_gameplay = (
            int(change_report.get("changed_count", 0)) > 0
            and not bool(change_report.get("border_only"))
        )
        if moved_gameplay:
            memory.pop(name, None)
            return
        memory[name] = memory.get(name, 0) + 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --locked --extra dev pytest tests/test_tool_agent_effect_memory.py -v`
Expected: PASS — 6 passed

- [ ] **Step 5: Call it from `_handle_action`**

In `_handle_action`, immediately after the `change_report` block added in Task 2, add:

```python
            self._record_action_effect(
                str(compact_payload.get("action_display") or ""),
                change_report,
                level_changed=bool(compact_payload.get("level_completed")),
            )
```

- [ ] **Step 6: Expose it to the sandbox and the prompt**

In `_serialized_runtime_state`, add one key to the returned dict:

```python
                "no_effect_actions": sorted(getattr(self, "_no_effect_actions", {})),
```

In `_SANDBOX_BOOTSTRAP`, inside `_refresh_state`, add alongside the other assignments:

```python
            runtime_globals["no_effect_actions"] = list(state_payload.get("no_effect_actions") or [])
```

`_world_model_status_lines` currently returns early when no model is saved, so the
dead-action line would be skipped in exactly the case it matters most. Replace the
whole method written in Task 5 Step 3 with this version:

```python
    def _world_model_status_lines(self) -> list[str]:
        """Render induced-model status for the prompt (safe on pre-upgrade pickles)."""
        source = getattr(self, "_world_model_source", "") or ""
        if not source.strip():
            lines = [
                "Induced world model: none saved yet. Write `encode(frame)` and "
                "`step(state, action)` as a source string and pass it to `save_model(source)`."
            ]
        else:
            lines = [
                f"Induced world model: saved, {len(source.splitlines())} lines, "
                "reloaded into every `python` call automatically."
            ]
            backtest = getattr(self, "_last_backtest", None)
            if isinstance(backtest, dict) and backtest.get("total"):
                status = "certified" if backtest.get("certified") else "not certified"
                lines.append(
                    f"Last backtest: {backtest.get('exact', 0)}/{backtest.get('total', 0)} "
                    f"transitions reproduced exactly ({status})."
                )
                mismatch = backtest.get("first_mismatch")
                if isinstance(mismatch, dict):
                    lines.append(
                        f"First mismatch at transition {mismatch.get('index')} "
                        f"on action {mismatch.get('action')}."
                    )
            else:
                lines.append(
                    "Last backtest: never run. Call `run_backtest()` before trusting this model."
                )
        dead_actions = sorted(getattr(self, "_no_effect_actions", {}))
        if dead_actions:
            lines.append(
                "Actions that changed nothing on this level so far: "
                + ", ".join(dead_actions)
                + ". Do not re-test them without a reason."
            )
        return lines
```

- [ ] **Step 7: Add a test for the merged status lines**

Append to `tests/test_tool_agent_world_model_prompt.py`:

```python
def test_status_lists_dead_actions():
    agent = _agent()
    agent._world_model_source = ""
    agent._last_backtest = None
    agent._no_effect_actions = {"LEFT": 3, "SPACE": 1}
    lines = agent._world_model_status_lines()
    assert any("LEFT, SPACE" in line for line in lines)
```

- [ ] **Step 8: Verify everything passes**

Run: `make prepare-ci`
Expected: ruff clean, all tests pass

- [ ] **Step 9: Commit**

```bash
git add inference/agent/tool_agent.py inference/agent/python_tool_sandbox.py tests/test_tool_agent_effect_memory.py tests/test_tool_agent_world_model_prompt.py && git commit -m "Remember actions that changed nothing on the current level"
```

---

## Closing out: the go/no-go measurement

This is not a code task. It is the reason the plan stops here.

- [ ] **Run the baseline and the candidate at >= 20 passes each**

```bash
make interactive GAME=[] GAME_TAGS=official SIMULATE_COMPETITION_ARCADE=true COMPETITION_CLONE_RUNS=110 N_PASSES=1 EXPERIMENTS_DIR=runs
```

- [ ] **Count certification, not score**

RHAE is too noisy at this stage — the baseline is 1.6002 ± 0.4475. Count the leading indicator instead, from the transcript line added in Task 5 Step 8:

```bash
grep -rho "certified=True" runs/<run-dir> | wc -l
```

Per game, record: was a model ever saved, what was the best `exact/total`, and did it ever certify.

- [ ] **Decide, and write the decision down**

- Certifies on **>= 3 of 25 games** → proceed to Phase 2 (`run_bfs`) and Phase 3 (`commit_actions`), each in its own plan.
- Certifies on **fewer than 3** → stop. The 27B cannot induce models reliably, so Phases 2–4 have no foundation. Move Phase 5 (trace distillation, `make traces` → SFT) to the critical path.

- [ ] **Only then run `make significance`**

```bash
make significance BASELINE_SCORE=docs/current-best-score.json CANDIDATE_SCORE=docs/candidate-score.json
```

Expect no significant RHAE movement from this plan alone. Phase 1 builds the foundation; the efficiency step-change arrives with Phase 2's search and Phase 3's gated commit. Treat a flat score here as expected, not as failure — and treat a *drop* as a real signal that the induction phase is eating the turn budget, which is what the §3 fallback in `improvment-v2.md` exists to catch.

---

## Deferred to later plans

| Item | Why not now |
| --- | --- |
| `run_bfs()` | needs a certified model to search inside; also needs a budget decision (30s sandbox cap vs host-side tool) |
| `commit_actions()` with abort-on-misprediction | needs `step()` to predict against; touches `solver.py`'s action path |
| Representation-revision escalation | needs observed backtest-failure patterns to tune the trigger |
| Click-target pruning from segmentation | the model can already derive this from `segmentation`; lower priority than harness-only capabilities |
| Oscillation guard | largely subsumed by no-effect memory plus planning |
| Trace distillation | separate subsystem — a training pipeline, not a harness change |
| Enforced certification fallback (K rounds / T seconds) | `improvment-v2.md` §3 requires this, but it is a fallback *from* the Phase 2 planning gate, which does not exist yet. Nothing in Phase 1 blocks the agent from acting, so there is nothing to fall back from. It must ship in the same plan as the gate, never after it. |
