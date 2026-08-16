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


def test_a_stored_model_that_fails_to_reload_reports_the_error():
    outcome = _run("result = world_model_error", world_model_source="raise ValueError('stale model')\n")
    assert "stale model" in outcome["world_model_error"]
    assert outcome["world_model_source"] == ""
