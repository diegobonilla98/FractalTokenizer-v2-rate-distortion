import numpy as np


def fixed_grid_assignment(
    grid_height: int,
    grid_width: int,
    token_rows: int,
    token_columns: int,
) -> tuple[np.ndarray, int]:
    if grid_height % token_rows or grid_width % token_columns:
        raise ValueError("token grid must divide the base grid exactly")
    y = np.arange(grid_height) * token_rows // grid_height
    x = np.arange(grid_width) * token_columns // grid_width
    assignment = y[:, None] * token_columns + x[None, :]
    return assignment.astype(np.uint16), token_rows * token_columns
