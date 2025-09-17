# pylint: disable=too-many-branches,too-many-statements
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt


def psi(h, w, K=64):
    """Implements the formula: Ψ(h,w) := ((h - 1) mod √K) · √K + ((w - 1) mod √K).

    For an 8x8 grid, K=64, so √K = 8
    h and w use 1-based indexing as specified.

    Args:
        h: Row index (1-based).
        w: Column index (1-based).
        K: Total number of elements in the grid (default: 64).

    Returns:
        Computed psi value according to the formula.
    """
    sqrt_K = int(np.sqrt(K))
    # Using 1-based indexing with (h-1) and (w-1)
    return ((h - 1) % sqrt_K) * sqrt_K + ((w - 1) % sqrt_K)


def _create_grid_values(grid_size, K):
    """Create grid values using 1-based indexing."""
    grid_values = np.zeros((grid_size, grid_size))
    for h in range(1, grid_size + 1):
        for value_w in range(1, grid_size + 1):
            grid_values[h - 1, value_w - 1] = psi(h, value_w, K)
    return grid_values


def _create_color_mapping(grid_values):
    """Create color mapping for unique grid values."""
    unique_values = np.unique(grid_values)
    color_list = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
        "magenta",
        "yellow",
        "lime",
        "indigo",
        "navy",
        "teal",
    ]
    return {val: color_list[i % len(color_list)] for i, val in enumerate(unique_values)}


def _draw_grid_cells(ax, grid_values, value_to_color, grid_size):
    """Draw grid cells with colors and numbers."""
    for h in range(grid_size):
        for value_w in range(grid_size):
            value = grid_values[h, value_w]
            rect = patches.Rectangle(
                (value_w, grid_size - h - 1),
                1,
                1,
                linewidth=2,
                edgecolor="black",
                facecolor=value_to_color[value],
                alpha=0.3,
            )
            ax.add_patch(rect)
            ax.text(
                value_w + 0.5,
                grid_size - h - 0.5,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=32,
                fontweight="bold",
            )


def _setup_plot_axes(ax, grid_size, K):
    """Setup plot axes, ticks, and labels."""
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xticks([i + 0.5 for i in range(grid_size)], minor=True)
    ax.set_yticks([i + 0.5 for i in range(grid_size)], minor=True)
    ax.set_xticklabels([str(i) for i in range(1, grid_size + 1)], minor=True, fontsize=18)
    ax.set_yticklabels([str(i) for i in range(grid_size, 0, -1)], minor=True, fontsize=18)
    ax.tick_params(which="major", labelbottom=False, labelleft=False)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("$w$ (column index, 1-based)", fontsize=28, labelpad=15)
    ax.set_ylabel("$h$ (row index, 1-based)", fontsize=28, labelpad=15)
    ax.set_title(f"$K = {K}$", fontsize=48, pad=20)


def _get_expected_block_values(block_h, block_value_w, block_size, K):
    """Calculate expected values for a complete block."""
    expected_values = set()
    for h_offset in range(block_size):
        for w_offset in range(block_size):
            theoretical_h = block_h + h_offset + 1
            theoretical_value_w = block_value_w + w_offset + 1
            expected_values.add(psi(theoretical_h, theoretical_value_w, K))
    return expected_values


def _draw_block_border(ax, block_h, block_value_w, actual_height, actual_width, grid_size):
    """Draw border lines for a block."""
    ax.plot(
        [block_value_w, block_value_w + actual_width],
        [grid_size - block_h, grid_size - block_h],
        "k-",
        linewidth=5,
    )
    ax.plot(
        [block_value_w, block_value_w + actual_width],
        [grid_size - block_h - actual_height, grid_size - block_h - actual_height],
        "k-",
        linewidth=5,
    )
    ax.plot(
        [block_value_w, block_value_w],
        [grid_size - block_h - actual_height, grid_size - block_h],
        "k-",
        linewidth=5,
    )
    ax.plot(
        [block_value_w + actual_width, block_value_w + actual_width],
        [grid_size - block_h - actual_height, grid_size - block_h],
        "k-",
        linewidth=5,
    )


def _draw_block_borders(ax, grid_values, grid_size, K):
    """Draw borders around blocks that contain all unique values."""
    sqrt_K = int(np.sqrt(K))
    block_size = sqrt_K
    for block_h in range(0, grid_size, block_size):
        for block_value_w in range(0, grid_size, block_size):
            actual_h_end = min(block_h + block_size, grid_size)
            actual_w_end = min(block_value_w + block_size, grid_size)
            expected_values = _get_expected_block_values(block_h, block_value_w, block_size, K)
            if len(expected_values) == K:
                actual_height = actual_h_end - block_h
                actual_width = actual_w_end - block_value_w
                _draw_block_border(
                    ax, block_h, block_value_w, actual_height, actual_width, grid_size
                )


def _print_grid_info(grid_values, grid_size, K, filename):
    """Print grid information and statistics."""
    print(f"Grid saved as '{filename}'")  # noqa: WPS421
    print(r"\nGrid values (Ψ(h,w) for each position):")  # noqa: WPS421
    sqrt_K = int(np.sqrt(K))
    print(  # noqa: WPS421
        f"Formula: Ψ(h,w) = ((h - 1) mod {sqrt_K}) × " f"{sqrt_K} + ((w - 1) mod {sqrt_K})"
    )
    print(r"Rows (h) go from top to bottom, Columns (w) go from left to right\n")  # noqa: WPS421
    for h in range(1, grid_size + 1):
        row_values = [int(psi(h, grid_w, K)) for grid_w in range(1, grid_size + 1)]
        print(f"Row {h} ((h-1) mod {sqrt_K} = {(h - 1) % sqrt_K}): {row_values}")  # noqa: WPS421


def _print_examples(K):
    """Print example calculations."""
    print(r"\nExample calculations:")  # noqa: WPS421
    sqrt_K = int(np.sqrt(K))
    examples = [(1, 1), (1, 8), (8, 1), (8, 8), (4, 5)]
    for h, example_w in examples:
        value = psi(h, example_w, K)
        print(  # noqa: WPS421
            f"Ψ({h},{example_w}) = (({h}-1) mod {sqrt_K}) × {sqrt_K} + "
            f"(({example_w}-1) mod {sqrt_K}) = {(h - 1) % sqrt_K} × {sqrt_K} + "
            f"{(example_w - 1) % sqrt_K} = {value:.0f}"
        )


def _print_ck_sets(grid_size, K):
    """Print C_k sets (cells with the same Ψ value)."""
    print(r"\nSets C_k (cells with the same Ψ value):")  # noqa: WPS421
    value_to_cells = {}
    for h in range(1, grid_size + 1):
        for set_w in range(1, grid_size + 1):
            value = int(psi(h, set_w, K))
            if value not in value_to_cells:
                value_to_cells[value] = []
            value_to_cells[value].append((h, set_w))
    for k in sorted(value_to_cells.keys()):
        print(f"C_{k} = {value_to_cells[k]}")  # noqa: WPS421


def create_numbered_grid(K, grid_size=8, filename="numbered_grid.png"):
    """Creates an 8x8 grid where each square is numbered according to formula.

    Ψ(h,w) = ((h - 1) mod √K) · √K + ((w - 1) mod √K)
    Uses 1-based indexing for the formula.

    Args:
        K: Total number of elements in the grid.
        grid_size: Size of the grid (default: 8).
        filename: Output filename for the plot (default: "numbered_grid.png").
    """
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    grid_values = _create_grid_values(grid_size, K)
    value_to_color = _create_color_mapping(grid_values)
    _draw_grid_cells(ax, grid_values, value_to_color, grid_size)
    _setup_plot_axes(ax, grid_size, K)
    _draw_block_borders(ax, grid_values, grid_size, K)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    _print_grid_info(grid_values, grid_size, K, filename)
    _print_examples(K)
    _print_ck_sets(grid_size, K)


# Create and save the grid
if __name__ == "__main__":
    create_numbered_grid(K=4, filename="rank4checkerboard.png")
    create_numbered_grid(K=9, filename="rank9checkerboard.png")
    create_numbered_grid(K=16, filename="rank16checkerboard.png")
