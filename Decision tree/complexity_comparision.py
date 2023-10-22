import matplotlib.pyplot as plt
import numpy as np

# O(1) - Constant Time Complexity
def constant_time(n):
    return np.ones_like(n)  # Return an array of ones with the same shape as n

# O(n) - Linear Time Complexity
def linear_time(n):
    return n

# O(log n) - Logarithmic Time Complexity
def logarithmic_time(n):
    return np.log2(n)

# O(n log n) - Linearithmic Time Complexity
def linearithmic_time(n):
    return n * np.log2(n)

# O(nm log n) - Decision Tree Time Complexity
def nmlogn_time(n, m):
    return n * m * np.log2(n)

# O(n^2) - Quadratic Time Complexity
def quadratic_time(n):
    return n**2

n = np.arange(1, 200)
m1 = 2
m2 = 5

# Calculate the number of operations for each complexity
operations_constant = constant_time(n)
operations_linear = linear_time(n)
operations_logarithmic = logarithmic_time(n)
operations_linearithmic = linearithmic_time(n)
operations_nmlogn1 = nmlogn_time(n, m1)
operations_nmlogn2 = nmlogn_time(n, m2)
operations_quadratic = quadratic_time(n)

plt.plot(n, operations_constant, label="O(1)")
plt.plot(n, operations_linear, label="O(n)")
plt.plot(n, operations_logarithmic, label="O(log n)")
plt.plot(n, operations_linearithmic, label="O(n log n)")
plt.plot(n, operations_nmlogn1, label="Decision tree O(nm log n), m=2")
plt.plot(n, operations_nmlogn2, label="Decision tree O(nm log n), m=5")
plt.plot(n, operations_quadratic, label="O(n^2)")

plt.xlabel("Input Size (n)")
plt.ylabel("Number of Operations")
plt.title("Time complexity")
plt.legend()
plt.text(
    0.9,
    0.1,
    "Roll: 18, 25",
    ha="right",
    va="bottom",
    transform=plt.gca().transAxes,
    color="red",
    fontsize=12,
)
plt.savefig("complexity.png", dpi=300)
plt.show()
