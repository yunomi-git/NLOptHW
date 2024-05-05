## Problem 5
import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    lm = 1.0
    lr = 2.0 * lm
    dim = 2.0

    # Initialize
    a1 = 0.0
    a2 = 0.0
    w = 0.1
    x1 = w
    x2 = w

    max_iterations = 1000
    tolerance = 1e-4
    w_history = [w]
    a1_history = [a1]
    a2_history = [a2]

    for _ in range(max_iterations):
        # Update lagrange multipliers
        a1 = a1 - lr * (w - x1)
        a2 = a2 - lr * (w - x2)

        # Solve individual subproblems
        x1 = (2.0 * lm * w - a1 + 4.0) / (2.0 + 2.0 * lm)
        x2 = (2.0 * lm * w - a2 - 8.0) / (6.0 + 2.0 * lm)

        # Update global problem
        w_prev = w
        w = (x1 + x2) / dim + (a1 + a2) / dim / (2.0 * lm)

        # Bookkeeping
        w_history.append(w)
        a1_history.append(a1)
        a2_history.append(a2)
        if np.abs(w - w_prev) < tolerance:
            break


    plt.subplot(3, 1, 1)
    plt.plot(a1_history)
    plt.ylabel("a1")
    plt.subplot(3, 1, 2)
    plt.plot(a2_history)
    plt.ylabel("a2")
    plt.subplot(3, 1, 3)
    plt.plot(w_history)
    plt.ylabel("w")
    plt.show()
