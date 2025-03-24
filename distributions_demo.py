"""Utilize scipy.stats to plot famous distributions"""

import numpy as np
from scipy.stats import norm


def dist_normal(idv: np.ndarray, loc: float = 0, scale: float = 1) -> np.ndarray:
    """Generate Gaussian distribution."""

    return norm.pdf(idv, loc, scale)


if __name__ == "__main__":
    x = np.arange(-10, 10, 0.1)
    y = dist_normal(x, 2, 3)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, "r-", lw=5, alpha=0.6, label="Normal Distribution")
    ax.set_title("Some famous dist.")
    ax.legend()
    plt.show()
