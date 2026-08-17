from inference.agent.prompts import (
    STRUCTURED_RUNTIME_STATE_ADDENDUM,
    SIMULATOR_ADDENDUM,
)
from inference.agent.tool_agent import _build_system_prompt


def test_runtime_state_addendum_documents_no_effect_actions():
    assert "`no_effect_actions`" in STRUCTURED_RUNTIME_STATE_ADDENDUM


def test_runtime_state_addendum_documents_the_change_report_keys():
    assert "`change_report`" in STRUCTURED_RUNTIME_STATE_ADDENDUM
    for key in ("changed_count", "bounding_box", "border_only"):
        assert f"`{key}`" in STRUCTURED_RUNTIME_STATE_ADDENDUM


def test_runtime_state_addendum_warns_that_border_only_is_false_with_no_change():
    assert "`border_only` is `False` when nothing changed at all" in STRUCTURED_RUNTIME_STATE_ADDENDUM


def test_simulator_addendum_attributes_border_only_to_the_whole_call():
    assert "the last action" not in SIMULATOR_ADDENDUM
    assert "`action(...)` call" in SIMULATOR_ADDENDUM


def test_system_prompt_has_no_world_model_terminology_collision():
    """`save_model` must not survive the rename anywhere in the system prompt.

    The old prose-vs-executable-feature name collision caused the model to
    never call the sandbox tools in a real run; the fix renamed the feature
    to `save_simulator` and folded it into the prescribed default loop.
    """
    prompt = _build_system_prompt(tool_output_tokens=1024)
    assert "save_simulator" in prompt
    assert "save_model" not in prompt
    assert "The default loop is" in prompt
    assert "save_simulator" in prompt.split("The default loop is")[1].split("\n")[0]
