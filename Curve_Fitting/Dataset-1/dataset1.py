import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("dataset1.txt")
x = data[:, 0]                                  # Extracting data from dataset
y = data[:, 1]

M = np.column_stack([x, np.ones(len(x))])
(p1, p2), _, _, _ = np.linalg.lstsq(M, y, rcond=None)           # Calling lstsq function to obtain line parameters
print(f"The estimated equation is {p1} x + {p2}")

def stline(x, m, c):
    return m * x + c
y_est = stline(x, p1, p2)

plt.plot(x, y, label = "Given plot", color = "cyan")
plt.plot(x, y_est, label = "Estimated line", color = "blue")            # plotting curves and saving them
plt.errorbar(x[::25], y[::25], np.std(y - y_est), fmt='ro', label = "Errorbar" )
plt.legend(title = "Plot of x v/s y")
plt.savefig("dataset1.png")