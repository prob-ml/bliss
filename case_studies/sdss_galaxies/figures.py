# %%
import numpy as np
from matplotlib import pyplot as plt

# %%
# scatter plot of miscclassification probs
prob_galaxy = np.zeros((10,))
misclass = np.zeros((10,))
true_mags = np.zeros((10,))
probs_correct = prob_galaxy[~misclass]
probs_misclass = prob_galaxy[misclass]

plt.scatter(true_mags[~misclass], probs_correct, marker="x", c="b")
plt.scatter(true_mags[misclass], probs_misclass, marker="x", c="r")
plt.axhline(0.5, linestyle="--")
plt.axhline(0.1, linestyle="--")
plt.axhline(0.9, linestyle="--")

uncertain = (prob_galaxy[misclass] > 0.2) & (prob_galaxy[misclass] < 0.8)
r_uncertain = sum(uncertain) / len(prob_galaxy[misclass])
print(
    f"ratio misclass with probability between 10%-90%: {r_uncertain:.3f}",
)
