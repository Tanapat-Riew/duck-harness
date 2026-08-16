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
