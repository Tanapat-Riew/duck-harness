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


def test_status_reports_a_discarded_model_after_reload_failure():
    agent = _agent()
    agent._world_model_source = ""
    agent._last_backtest = None
    agent._world_model_error = "NameError: name 'foo' is not defined"
    lines = agent._world_model_status_lines()
    assert any("failed to reload" in line and "discarded" in line for line in lines)


def test_a_discarded_model_replaces_the_none_saved_yet_line():
    agent = _agent()
    agent._world_model_source = ""
    agent._last_backtest = None
    agent._world_model_error = "NameError: name 'foo' is not defined"
    lines = agent._world_model_status_lines()
    assert not any("none saved yet" in line for line in lines)


def test_status_lists_dead_actions():
    agent = _agent()
    agent._world_model_source = ""
    agent._last_backtest = None
    agent._no_effect_actions = {"LEFT": 3, "SPACE": 1}
    lines = agent._world_model_status_lines()
    assert any("LEFT, SPACE" in line for line in lines)
