# Final Problem 3

import numpy as np
from typing import List
import matplotlib.pyplot as plt

class Constraint:
    def __init__(self, a: np.ndarray, b):
        self.a = a
        self.b = b

    def compute(self, x: np.ndarray):
        return np.dot(self.a, x) - self.b

def compute_gradient(constraints: List[Constraint], x):
    gradient = np.zeros(2)
    for constraint in constraints:
        gradient += - constraint.a / constraint.compute(x)
    return gradient

def compute_hessian(constraints: List[Constraint], x):
    hessian = np.zeros((2,2))
    for constraint in constraints:
        hessian += np.outer(constraint.a, constraint.a) / (constraint.compute(x) ** 2)
    return hessian

if __name__=="__main__":
    x0 = np.array([0.5, 0.5])
    nu0 = 1
    cp = 4
    beta = 1 - 1.0 / (8 * np.sqrt(cp))

    constraints = [Constraint(np.array([-2, 1]), 0), # 0,0 to 1,2
                   Constraint(np.array([0, 1]), 0),  # 0,0 to 2,0
                   Constraint(np.array([-1, -1]), -3), # 1,2 to 2,1
                   Constraint(np.array([-1, 0]), -2)   # 2,0 to 2,1
                   ]
    x = x0
    nu = nu0

    starting_gradient = compute_gradient(constraints, x0)

    max_steps = 300
    x_history = np.zeros((max_steps, 2))
    for i in range(300):
        hessian = compute_hessian(constraints, x)
        gradient = compute_gradient(constraints, x)

        nu = nu * beta
        x = x - np.linalg.solve(hessian, -nu * starting_gradient + gradient)
        x_history[i] = x

    print(x)
    plt.plot([0, 2, 2, 1, 0], [0, 0, 1, 2, 0])
    plt.plot(x_history[:, 0], x_history[:, 1])
    plt.scatter(x[0], x[1])
    plt.show()
