import itertools

import torch
from matplotlib import pyplot as plt

n = 2
binary_combinations = list(itertools.product([0, 1], repeat=n * n))
mask_patterns = torch.tensor(binary_combinations).view(-1, n, n)

# Create a figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=15, figsize=(15, 2))

# Iterate over each 2x2 matrix and plot it
for i in range(15):
    ax = axes[i]
    ax.imshow(mask_patterns[i], cmap="gray", interpolation="nearest")

    # Hide axis labels but keep borders
    ax.set_xticks([])
    ax.set_yticks([])

    # Add iterate number above subplot
    ax.set_title(f"{i + 1}")

    # Add a border around the subplot
    for spine in ax.spines.values():
        spine.set_edgecolor("red")
        spine.set_linewidth(2)
        spine.set_visible(True)  # Ensure the border is visible

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

history_mask = mask_patterns[5].repeat([28, 28])
plt.imshow(history_mask, cmap="gray", interpolation="nearest")
