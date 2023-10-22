import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def plot_transformed_circle(mat):
    theta = np.linspace(0, 2 * np.pi, 100)
    eigenvalues, eigenvectors = np.linalg.eig(mat)

    # Generate the x and y coordinates of the unit circle
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))

    ax1.plot(x_circle, y_circle, label="Original Circle")
    ax1.set_title("Before transformation")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.axhline(0, color="black")
    ax1.axvline(0, color="black")
    ax1.text(
        0.9,
        0.1,
        "",
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=14,
    )
    ax1.grid()

    for eigenvector in eigenvectors.T:
        scaled_eigenvector = np.dot(mat, eigenvector)
        ax1.arrow(
            0,
            0,
            scaled_eigenvector[0],
            scaled_eigenvector[1],
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
        )

    # Matrix multiplication of circle with matrix
    x_transformed = mat[0, 0] * x_circle + mat[0, 1] * y_circle
    y_transformed = mat[1, 0] * x_circle + mat[1, 1] * y_circle

    ax2.plot(x_transformed, y_transformed, label="Transformed Circle")
    ax2.set_title("After transformation")
    ax2.axhline(0, color="black")
    ax2.axvline(0, color="black")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid()

    for eigenvector in eigenvectors.T:
        scaled_eigenvector = np.dot(mat, eigenvector)
        ax2.arrow(
            0,
            0,
            scaled_eigenvector[0],
            scaled_eigenvector[1],
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
        )
        ax2.text(
            0.9,
            0.1,
            "",
            ha="right",
            va="bottom",
            transform=plt.gca().transAxes,
            fontsize=14,
        )
        
    plt.show()

mat1 = np.random.rand(2, 2)
plot_transformed_circle(mat1)


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
        "",
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

