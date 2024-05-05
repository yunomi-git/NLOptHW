## Problem 6
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
    x0 = np.array([0.2, 0.2])
    # For 6.1
    # constraints = [Constraint(np.array([1, 0]), 0), # 0,0 to 0,1
    #                Constraint(np.array([0, 1]), 0), # 0,0 to 1,0
    #                Constraint(np.array([-1, -1]), -1)  # 1,0 to 0,1
    #                ]

    # For 6.2
    constraints = [Constraint(np.array([-2, 1]), 0), # 0,0 to 1,2
                   Constraint(np.array([0, 1]), 0),  # 0,0 to 2,0
                   Constraint(np.array([-1, -1]), -3), # 1,2 to 2,1
                   Constraint(np.array([-1, 0]), -2)   # 2,0 to 2,1
                   ]


    x = x0
    for i in range(15):
        gradient = compute_gradient(constraints, x)
        hessian = compute_hessian(constraints, x)
        inv_hessian = np.linalg.inv(hessian)

        lm = np.sqrt(gradient.T @ inv_hessian @ gradient)
        lr = 1.0 / (1 + lm)

        x = x - lr * inv_hessian @ gradient
        print(x)