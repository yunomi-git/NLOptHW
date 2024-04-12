import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np

n, d = 20, 2 # Example: 20 points in 2D space
points = np.random.rand(n, d)

######## write your code here ########
M = cp.Variable((d, d), PSD=True)
z = cp.Variable(d)
obj = cp.Minimize(-2 * cp.log_det(M))
# We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
soc_constraints = [
      cp.SOC(1, M @ points[i] - z) for i in range(n)
]
prob = cp.Problem(obj,
                  soc_constraints)
prob.solve()
M_sol = M.value
z_sol = z.value

A = M_sol @ M_sol
c = np.linalg.inv(M_sol) @ z_sol
######################################
def create_plots(A, c):
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(A)

    # Get the lengths of the semi-axes
    axes_lengths = 1 / np.sqrt(eigvals)

    # Generate ellipse points in canonical form
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = axes_lengths[0] * np.cos(theta)
    ellipse_y = axes_lengths[1] * np.sin(theta)

    # Rotate the points
    x_rotated = eigvecs[0, 0] * ellipse_x + eigvecs[0, 1] * ellipse_y
    y_rotated = eigvecs[1, 0] * ellipse_x + eigvecs[1, 1] * ellipse_y

    # Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(x_rotated + c[0], y_rotated + c[1], label='MVEE')
    plt.scatter(points[:, 0], points[:, 1], color='red', label='points')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Ellipse from General 2D Matrix A')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

# The following codes can create plots of the minimum volume enclosing ellipsoid, given A and c
create_plots(A, c)