import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

N = 20
np.random.seed(1234)
w = np.random.rand(N)
k = 10
L0 = 0.1
a, b = np.array([-1,0]), np.array([1,0])
e2 = np.array([0,1])


######## write your code here ########
x = cp.Variable((N, 2))
obj1 = cp.sum([w[i] * e2 @ x[i] for i in range(N)])
obj2 = k / 2.0 * cp.sum([
                      cp.square(
                          cp.maximum(
                              cp.norm(x[i] - x[i + 1]) - L0, 0
                          )
                      ) for i in range(N - 1)]
                      )
obj = cp.Minimize(obj1 + obj2)

constraints = [x[0] == a,
                x[N-1] == b]
prob = cp.Problem(obj,
                  constraints)
prob.solve()
######################################

# create plots
points = x.value
for i in range(len(points) - 1):
    segment_start = points[i]
    segment_end = points[i + 1]
    plt.plot([segment_start[0], segment_end[0]], [segment_start[1], segment_end[1]], marker='o')

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)
plt.axis('equal')
plt.show()