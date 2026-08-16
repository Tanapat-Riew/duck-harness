from inference.agent.tool_agent import ToolAgent

MODEL_A = "def encode(f):\n    return 0\n"
MODEL_B = "def encode(f):\n    return 1\n"
CERTIFIED = {"ok": True, "total": 2, "exact": 2, "certified": True, "first_mismatch": None}


def _agent():
    agent = ToolAgent.__new__(ToolAgent)
    agent._world_model_source = ""
    agent._world_model_error = ""
    agent._last_backtest = None
    agent._no_effect_actions = {}
    return agent


def _sandbox_result(source="", error="", backtest=None):
    return {
        "world_model_source": source,
        "world_model_error": error,
        "last_backtest": backtest,
    }


def test_saving_a_new_source_clears_a_previous_backtest():
    agent = _agent()
    agent._apply_sandbox_world_model_result(_sandbox_result(source=MODEL_A, backtest=CERTIFIED))
    agent._apply_sandbox_world_model_result(_sandbox_result(source=MODEL_B))
    assert agent._world_model_source == MODEL_B
    assert agent._last_backtest is None


def test_reloading_the_same_source_keeps_its_backtest():
    agent = _agent()
    agent._apply_sandbox_world_model_result(_sandbox_result(source=MODEL_A, backtest=CERTIFIED))
    agent._apply_sandbox_world_model_result(_sandbox_result(source=MODEL_A))
    assert agent._last_backtest == CERTIFIED


def test_a_reload_error_discards_the_model_and_its_backtest():
    agent = _agent()
    agent._apply_sandbox_world_model_result(_sandbox_result(source=MODEL_A, backtest=CERTIFIED))
    agent._apply_sandbox_world_model_result(_sandbox_result(error="NameError: nope"))
    assert agent._world_model_source == ""
    assert agent._last_backtest is None
    assert "NameError" in agent._world_model_error


def test_a_reload_error_stays_pending_across_a_later_call_without_one():
    agent = _agent()
    agent._apply_sandbox_world_model_result(_sandbox_result(error="NameError: nope"))
    agent._apply_sandbox_world_model_result(_sandbox_result())
    assert "NameError" in agent._world_model_error


def test_saving_a_fresh_model_resolves_a_pending_reload_error():
    agent = _agent()
    agent._apply_sandbox_world_model_result(_sandbox_result(error="NameError: nope"))
    agent._apply_sandbox_world_model_result(_sandbox_result(source=MODEL_B))
    assert agent._world_model_error == ""
    assert agent._world_model_source == MODEL_B


def test_a_pending_error_is_rendered_once_and_then_forgotten():
    agent = _agent()
    agent._apply_sandbox_world_model_result(_sandbox_result(error="NameError: nope"))
    first = agent._world_model_status_lines()
    second = agent._world_model_status_lines()
    assert any("failed to reload" in line for line in first)
    assert not any("failed to reload" in line for line in second)


def test_bookkeeping_survives_an_agent_pickled_before_these_attributes_existed():
    agent = ToolAgent.__new__(ToolAgent)
    agent._apply_sandbox_world_model_result(_sandbox_result(source=MODEL_A, backtest=CERTIFIED))
    assert agent._world_model_source == MODEL_A
    assert agent._last_backtest == CERTIFIED
