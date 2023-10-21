import matplotlib.pyplot as plt
3 from scipy.stats import gaussian_kde
4
5
6 def plot_scatter(x, y, title, label):
7 colors = np.random.rand(len(x))
8 plt.scatter(x, y, c=colors, label=label)
9 plt.xlim(-3, 3)
10 plt.ylim(-3, 3)
11 add_labels(title, label)
12 plt.show()
13
14
15 def add_labels(title, label):
16 plt.title(title)
17 plt.xlabel("X-axis")
18 plt.ylabel("Y-axis")
19 plt.axhline(0, color="black")
20 plt.axvline(0, color="black")
21 plt.grid()
22 plt.legend(loc="upper right", markerscale=0.7)
23 plt.text(
24 0.9,
25 0.1,
26 "Kshitiz poudel",
27 ha="right",
28 va="bottom",
29 transform=plt.gca().transAxes,
30 fontsize=14,
31 )
32
33
34 def contour_plot(x, axis_min, axis_max):
35 k = gaussian_kde([x[:, 0], x[:, 1]])
36 xi, yi = np.mgrid[axis_min:axis_max:100j,
axis_min:axis_max:100j]
37 zi = k(np.vstack([xi.flatten(), yi.flatten()])
)
38
39 contour = plt.contourf(xi, yi, zi.reshape(xi.
shape), cmap="viridis")
40
41 # Retrieve the contour lines
42 contour_lines = contour.collections[0]
43 plt.colorbar()
44 plt.xlim(-3, 3)
45 plt.ylim(-3, 3)
46 add_labels("2D Contour Plot", "")
