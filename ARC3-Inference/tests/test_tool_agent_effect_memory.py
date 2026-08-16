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
