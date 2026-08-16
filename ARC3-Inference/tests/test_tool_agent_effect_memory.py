from inference.agent.tool_agent import ToolAgent, _action_resets_effect_memory


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


def test_a_batch_of_actions_is_not_attributed_to_the_last_one():
    agent = _agent()
    agent._record_action_effect("RIGHT", _report(0), level_changed=False, executed_count=2)
    assert agent._no_effect_actions == {}


def test_a_positional_click_is_never_remembered():
    agent = _agent()
    agent._record_action_effect("MOUSE(row=3, col=4)", _report(0), level_changed=False)
    assert agent._no_effect_actions == {}


def test_a_level_completion_resets_the_memory():
    assert _action_resets_effect_memory({"level_completed": True}) is True


def test_a_game_over_resets_the_memory():
    assert _action_resets_effect_memory({"game_over": True}) is True


def test_a_run_completion_resets_the_memory():
    assert _action_resets_effect_memory({"run_complete": True}) is True


def test_an_ordinary_action_does_not_reset_the_memory():
    assert _action_resets_effect_memory({"board_changed": True, "executed": True}) is False
