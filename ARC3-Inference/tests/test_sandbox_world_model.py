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


def _initial_state(world_model_source="", history=None, no_effect_actions=None):
    if history is None:
        history = [
            {"action": "", "frame": _frame_payload(0, "x..")},
            {"action": "RIGHT", "frame": _frame_payload(1, ".x.")},
            {"action": "RIGHT", "frame": _frame_payload(2, "..x")},
        ]
    return {
        "current_frame": _frame_payload(2, "..x"),
        "history": history,
        "valid_actions": ["RIGHT"],
        "last_action_result": {},
        "world_model_source": world_model_source,
        "no_effect_actions": no_effect_actions,
    }


def _no_actions(actions):
    raise AssertionError("this test must not step the environment")


def _run(code, world_model_source="", history=None, no_effect_actions=None):
    return run_sandboxed_python(
        code=code,
        timeout_seconds=20,
        initial_state=_initial_state(
            world_model_source, history=history, no_effect_actions=no_effect_actions
        ),
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


def test_saving_a_good_model_clears_a_stale_reload_error():
    outcome = _run(
        f'result = save_model("""{CORRECT_MODEL}""")',
        world_model_source="raise ValueError('stale model')\n",
    )
    assert outcome["result"]["ok"] is True
    assert "def step(state, action):" in outcome["world_model_source"]
    assert not outcome["world_model_error"]


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
    assert report["total"] == 2
    assert report["exact"] == 0


def test_backtest_without_a_model_reports_not_ok():
    outcome = _run("result = run_backtest()")
    assert outcome["result"]["ok"] is False
    assert "save_model" in outcome["result"]["error"]


def test_backtest_with_zero_transitions_never_certifies():
    outcome = _run(
        "result = run_backtest()",
        world_model_source=CORRECT_MODEL,
        history=[{"action": "", "frame": _frame_payload(0, "x..")}],
    )
    report = outcome["result"]
    assert report["total"] == 0
    assert report["exact"] == 0
    assert report["certified"] is False


def test_backtest_skips_a_transition_with_no_before_frame():
    outcome = _run(
        "result = run_backtest()",
        world_model_source=CORRECT_MODEL,
        history=[
            {"action": "RIGHT", "frame": _frame_payload(1, ".x.")},
            {"action": "RIGHT", "frame": _frame_payload(2, "..x")},
        ],
    )
    report = outcome["result"]
    assert report["total"] == 1
    assert report["exact"] == 1
    assert report["certified"] is True


def test_backtest_with_max_transitions_zero_reports_no_transitions():
    outcome = _run("result = run_backtest(max_transitions=0)", world_model_source=CORRECT_MODEL)
    report = outcome["result"]
    assert report["total"] == 0
    assert report["certified"] is False


def test_no_effect_actions_are_exposed_to_the_sandbox():
    outcome = _run("result = no_effect_actions", no_effect_actions=["LEFT", "SPACE"])
    assert outcome["result"] == ["LEFT", "SPACE"]


def test_no_effect_actions_defaults_to_an_empty_list():
    outcome = _run("result = no_effect_actions")
    assert outcome["result"] == []
