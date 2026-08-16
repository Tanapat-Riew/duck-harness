from inference.agent.prompts import (
    STRUCTURED_RUNTIME_STATE_ADDENDUM,
    WORLD_MODEL_ADDENDUM,
)


def test_runtime_state_addendum_documents_no_effect_actions():
    assert "`no_effect_actions`" in STRUCTURED_RUNTIME_STATE_ADDENDUM


def test_runtime_state_addendum_documents_the_change_report_keys():
    assert "`change_report`" in STRUCTURED_RUNTIME_STATE_ADDENDUM
    for key in ("changed_count", "bounding_box", "border_only"):
        assert f"`{key}`" in STRUCTURED_RUNTIME_STATE_ADDENDUM


def test_runtime_state_addendum_warns_that_border_only_is_false_with_no_change():
    assert "`border_only` is `False` when nothing changed at all" in STRUCTURED_RUNTIME_STATE_ADDENDUM


def test_world_model_addendum_attributes_border_only_to_the_whole_call():
    assert "the last action" not in WORLD_MODEL_ADDENDUM
    assert "`action(...)` call" in WORLD_MODEL_ADDENDUM
