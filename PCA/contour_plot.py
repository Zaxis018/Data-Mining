import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def plot_scatter(x, y, title, label):
    colors = np.random.rand(len(x))
    plt.scatter(x, y, c=colors, label=label)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    add_labels(title, label)
    plt.show()

def add_labels(title, label):
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.grid()
    plt.legend(loc="upper right", markerscale=0.7)
    plt.text(
        0.9,
        0.1,
        "kshitiz poudel",
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=14,
    )

def contour_plot(x, axis_min, axis_max):
    k = gaussian_kde([x[:, 0], x[:, 1]])
    xi, yi = np.mgrid[axis_min:axis_max:100j, axis_min:axis_max:100j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    contour = plt.contourf(xi, yi, zi.reshape(xi.shape), cmap="viridis")

    # Retrieve the contour lines
    contour_lines = contour.collections[0]
    plt.colorbar()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    add_labels("2D Contour Plot", "")

