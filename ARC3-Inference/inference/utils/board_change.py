"""Classify what changed between two board grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

Grid = Sequence[Sequence[int]]

MAX_REPORTED_CELLS = 20


@dataclass(frozen=True)
class BoardChangeReport:
    """Summary of the cells that differ between two grids."""

    changed_count: int
    changed_cells: tuple[tuple[int, int], ...]
    bounding_box: tuple[int, int, int, int] | None
    border_only: bool

    def as_dict(self) -> dict[str, Any]:
        """Render the report as a compact JSON-safe dict for tool results."""
        return {
            "changed_count": self.changed_count,
            "changed_cells": [list(cell) for cell in self.changed_cells[:MAX_REPORTED_CELLS]],
            "bounding_box": list(self.bounding_box) if self.bounding_box is not None else None,
            "border_only": self.border_only,
        }


def _grid_shape(grid: Grid) -> tuple[int, int]:
    """Return (rows, cols) tolerating ragged rows."""
    return len(grid), max((len(row) for row in grid), default=0)


def diff_cells(before: Grid, after: Grid) -> tuple[tuple[int, int], ...]:
    """Return every (row, col) whose value differs, treating missing cells as None."""
    changed: list[tuple[int, int]] = []
    for row_index in range(max(len(before), len(after))):
        before_row = before[row_index] if row_index < len(before) else ()
        after_row = after[row_index] if row_index < len(after) else ()
        for col_index in range(max(len(before_row), len(after_row))):
            before_value = before_row[col_index] if col_index < len(before_row) else None
            after_value = after_row[col_index] if col_index < len(after_row) else None
            if before_value != after_value:
                changed.append((row_index, col_index))
    return tuple(changed)


def classify_board_change(before: Grid, after: Grid, *, border_width: int = 2) -> BoardChangeReport:
    """Diff two grids and report whether the change is confined to a border strip.

    ``border_only`` is the signal that an action moved only a HUD or timer bar
    rather than gameplay state. It is False when nothing changed at all.
    """
    changed = diff_cells(before, after)
    if not changed:
        return BoardChangeReport(0, (), None, False)
    before_rows, before_cols = _grid_shape(before)
    after_rows, after_cols = _grid_shape(after)
    rows = max(before_rows, after_rows)
    cols = max(before_cols, after_cols)
    row_values = [cell[0] for cell in changed]
    col_values = [cell[1] for cell in changed]
    bounding_box = (min(row_values), min(col_values), max(row_values), max(col_values))
    border_only = all(
        row < border_width
        or row >= rows - border_width
        or col < border_width
        or col >= cols - border_width
        for row, col in changed
    )
    return BoardChangeReport(len(changed), changed, bounding_box, border_only)
