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
