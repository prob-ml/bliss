import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def psi(h, w, K=64):
    """
    Implements the formula: Ψ(h,w) := ((h - 1) mod √K) · √K + ((w - 1) mod √K)

    For an 8x8 grid, K=64, so √K = 8
    h and w use 1-based indexing as specified.
    """
    sqrt_K = int(np.sqrt(K))
    # Using 1-based indexing with (h-1) and (w-1)
    return ((h - 1) % sqrt_K) * sqrt_K + ((w - 1) % sqrt_K)


def create_numbered_grid(K, grid_size=8, filename="numbered_grid.png"):
    """
    Creates an 8x8 grid where each square is numbered according to
    Ψ(h,w) = ((h - 1) mod √K) · √K + ((w - 1) mod √K)
    Uses 1-based indexing for the formula.
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create the grid values using 1-based indexing
    grid_values = np.zeros((grid_size, grid_size))
    for h in range(1, grid_size + 1):  # 1-based row index
        for w in range(1, grid_size + 1):  # 1-based column index
            # Store in 0-based array
            grid_values[h - 1, w - 1] = psi(h, w, K)

    # Create distinct colors for each unique value
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
    value_to_color = {val: color_list[i % len(color_list)] for i, val in enumerate(unique_values)}

    # Draw the grid
    for h in range(grid_size):
        for w in range(grid_size):
            # Get the value for this cell
            value = grid_values[h, w]

            # Draw the square with distinct color based on value
            rect = patches.Rectangle(
                (w, grid_size - h - 1),
                1,
                1,
                linewidth=2,
                edgecolor="black",
                facecolor=value_to_color[value],
                alpha=0.3,
            )
            ax.add_patch(rect)

            # Add the number text in the center of each square
            ax.text(
                w + 0.5,
                grid_size - h - 0.5,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=32,
                fontweight="bold",
            )

    # Set up the plot
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")

    # Add grid lines
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True, linewidth=0.5, alpha=0.5)

    # Add tick labels at cell centers for 1-based indexing
    ax.set_xticks([i + 0.5 for i in range(grid_size)], minor=True)
    ax.set_yticks([i + 0.5 for i in range(grid_size)], minor=True)
    ax.set_xticklabels([str(i) for i in range(1, grid_size + 1)], minor=True, fontsize=18)
    ax.set_yticklabels([str(i) for i in range(grid_size, 0, -1)], minor=True, fontsize=18)
    ax.tick_params(which="major", labelbottom=False, labelleft=False)
    ax.tick_params(which="minor", length=0)

    # Labels
    ax.set_xlabel("$w$ (column index, 1-based)", fontsize=28, labelpad=15)
    ax.set_ylabel("$h$ (row index, 1-based)", fontsize=28, labelpad=15)
    ax.set_title(f"$K = {K}$", fontsize=48, pad=20)

    # Add heavy borders around blocks that contain all unique values
    sqrt_K = int(np.sqrt(K))
    block_size = sqrt_K

    # Draw heavy borders for blocks (complete or partial)
    for block_h in range(0, grid_size, block_size):
        for block_w in range(0, grid_size, block_size):
            # Check if this block contains all unique values that would appear in a complete block
            block_values = set()
            actual_h_end = min(block_h + block_size, grid_size)
            actual_w_end = min(block_w + block_size, grid_size)

            for h in range(block_h, actual_h_end):
                for w in range(block_w, actual_w_end):
                    block_values.add(grid_values[h, w])

            # Calculate what values should be in this theoretical complete block
            expected_values = set()
            for h_offset in range(block_size):
                for w_offset in range(block_size):
                    # Use 1-based indexing for the formula
                    theoretical_h = block_h + h_offset + 1
                    theoretical_w = block_w + w_offset + 1
                    expected_values.add(psi(theoretical_h, theoretical_w, K))

            # If this block's pattern would contain all K unique values (even if partial), draw border
            if len(expected_values) == K:
                actual_height = actual_h_end - block_h
                actual_width = actual_w_end - block_w

                # Draw border lines for the actual block size (may be partial)
                # Top line
                ax.plot(
                    [block_w, block_w + actual_width],
                    [grid_size - block_h, grid_size - block_h],
                    "k-",
                    linewidth=5,
                )
                # Bottom line
                ax.plot(
                    [block_w, block_w + actual_width],
                    [grid_size - block_h - actual_height, grid_size - block_h - actual_height],
                    "k-",
                    linewidth=5,
                )
                # Left line
                ax.plot(
                    [block_w, block_w],
                    [grid_size - block_h - actual_height, grid_size - block_h],
                    "k-",
                    linewidth=5,
                )
                # Right line
                ax.plot(
                    [block_w + actual_width, block_w + actual_width],
                    [grid_size - block_h - actual_height, grid_size - block_h],
                    "k-",
                    linewidth=5,
                )

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Grid saved as '{filename}'")

    # Print the grid values for verification
    print("\\nGrid values (Ψ(h,w) for each position):")
    print(
        f"Formula: Ψ(h,w) = ((h - 1) mod {int(np.sqrt(K))}) × {int(np.sqrt(K))} + ((w - 1) mod {int(np.sqrt(K))})"
    )
    print("Rows (h) go from top to bottom, Columns (w) go from left to right\\n")
    sqrt_K = int(np.sqrt(K))
    for h in range(1, grid_size + 1):
        row_values = [int(psi(h, w, K)) for w in range(1, grid_size + 1)]
        print(f"Row {h} ((h-1) mod {sqrt_K} = {(h-1) % sqrt_K}): {row_values}")

    # Print some example calculations
    print("\\nExample calculations:")
    examples = [(1, 1), (1, 8), (8, 1), (8, 8), (4, 5)]
    for h, w in examples:
        value = psi(h, w, K)
        print(
            f"Ψ({h},{w}) = (({h}-1) mod {sqrt_K}) × {sqrt_K} + (({w}-1) mod {sqrt_K}) = {(h-1)%sqrt_K} × {sqrt_K} + {(w-1)%sqrt_K} = {value:.0f}"
        )

    # Show which cells belong to each C_k set
    print("\\nSets C_k (cells with the same Ψ value):")
    value_to_cells = {}
    for h in range(1, grid_size + 1):
        for w in range(1, grid_size + 1):
            value = int(psi(h, w, K))
            if value not in value_to_cells:
                value_to_cells[value] = []
            value_to_cells[value].append((h, w))

    for k in sorted(value_to_cells.keys()):
        print(f"C_{k} = {value_to_cells[k]}")


# Create and save the grid
if __name__ == "__main__":
    create_numbered_grid(K=4, filename="rank4checkerboard.png")
    create_numbered_grid(K=9, filename="rank9checkerboard.png")
    create_numbered_grid(K=16, filename="rank16checkerboard.png")
