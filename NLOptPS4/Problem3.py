import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

def enforce_positive(x):
    x[x<0] = 0
    return x

def get_gradient(A, b, x):
    return 2 * A.T @ (A @ x - b)

def get_objective(A, b, x):
    # return 0.5 * np.square(np.linalg.norm(A @ x - b))
    return np.linalg.norm(A @ np.maximum(x, 0) - b) ** 2 / 2

#######################################################################
# Question 1
def optimize_cvxpy(A, b, n):
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.square(cp.norm(A @ x - b)))
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    constraints = [x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(obj,
                      constraints)
    prob.solve()
    x_sol = np.array(x.value)
    return x_sol
    # print(x_sol)

#######################################################################
# Question 2
def find_bounds(func, lower_x, upper_x):
    # For decreasing function
    upper_val = func(upper_x)
    while upper_val > 0:
        prev_lower_x = lower_x
        lower_x = upper_x
        upper_x += 2*(upper_x - prev_lower_x)
        upper_val = func(upper_x)
    lower_val = func(lower_x)
    while lower_val < 0:
        prev_upper_x = upper_x
        upper_x = lower_x
        lower_x -= 2*(prev_upper_x - lower_x)
        lower_val = func(upper_x)
    return lower_x - 0.1, upper_x + 0.1

def bisection_search(func, lower_x=-1.0, upper_x=1.0, tolerance=1e-8, max_steps = 200):
    lower_x, upper_x = find_bounds(func, lower_x, upper_x)
    # for decreasing functions
    test_x = (lower_x + upper_x) / 2.0
    test_val = func(test_x)
    step = 0
    while np.abs(test_val) > tolerance and step < max_steps:
        if test_val > 0:
            lower_x = test_x
        else:
            upper_x = test_x
        test_x = (lower_x + upper_x) / 2.0
        test_val = func(test_x)
        step += 1
    return test_x

def proj_simplex(x):
    # Project point x onto simplex
    def get_simplex_constraint_function(c):
        def func(lm):
            return np.dot(np.ones(c.shape),
                          enforce_positive(c - lm)
                          ) - 1
        return func

    func = get_simplex_constraint_function(x)
    lm = bisection_search(func)
    proj = enforce_positive(x - lm)
    return proj

# Question 3
def project_simplex_xlogx(x, learning_rate, gradient):
    proj = x * np.exp(-learning_rate * gradient)
    proj = enforce_positive(proj)
    proj /= np.sum(proj)
    return proj


# Question 2, 3
def optimize_gradient_descent(A, b, x0, steps, learning_rate, proj_method="euclid"):
    # methods: euclid, log
    obj_history = [get_objective(A, b, x0)]
    x = x0
    for i in range(steps):
        gradient = get_gradient(A, b, x)
        x = x - learning_rate * gradient
        # Project
        if proj_method == "euclid": # Question 2
            x = proj_simplex(x)
        else:
            x = project_simplex_xlogx(x, learning_rate, gradient) # Question 3
        obj_history.append(get_objective(A, b, x))
    # print(x)
    return np.array(obj_history)

############################################################
# Question 4
def optimize_nesterov_accelerated(A, b, x0, steps, learning_rate):
    obj_history = [get_objective(A, b, x0)]
    x_1 = x0
    x = x0
    for t in range(1, steps+1):
        y = x + (t-2) / (t+1) * (x - x_1)
        gradient = get_gradient(A, b, y)
        y = y - learning_rate * gradient
        # Project
        y = proj_simplex(y)
        # Save
        x_1 = x
        x = y
        obj_history.append(get_objective(A, b, x))
    # print(x)
    return np.array(obj_history)

##################################################################
# Question 5
def optimize_ADMM(A, b, x0, steps, rho, tau):
    obj_history = [get_objective(A, b, x0)]
    # initialize all variables
    x = x0
    z = x0
    y = np.zeros(x.shape)

    for i in range(steps):
        # First solve x
        x = np.linalg.inv(A.T @ A + rho * np.eye(len(x0))) @ (A.T @ b + rho * z - y)
        # Then update z
        z = proj_simplex(x + y / rho)
        # Then update lagrange multipliers
        y = y + tau * rho * (x - z)
        obj_history.append(get_objective(A, b, z))
    # print(x)
    return np.array(obj_history)

def debug_project_simplex():
    point = np.array([200, 10, 10])
    x = proj_simplex(point)
    x_log = project_simplex_xlogx(point, 0.1, 1/point)
    print(x)
    print(x_log)

if __name__=="__main__":
    np.set_printoptions(precision=5)
    m, n = 500, 200
    np.random.seed(2024)

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    x0 = np.ones(n)/n

    steps = 30
    lr = 0.2e-3

    history_accel = optimize_nesterov_accelerated(A, b, x0, steps=steps, learning_rate=lr)
    history_proj = optimize_gradient_descent(A, b, x0, steps=steps, learning_rate=lr, proj_method="euclid")
    history_log = optimize_gradient_descent(A, b, x0, steps=steps, learning_rate=lr, proj_method="log")
    history_admm = optimize_ADMM(A, b, x0, steps=steps, rho=10, tau=1.0)

    opt_x = optimize_cvxpy(A, b, n)
    f_opt = get_objective(A, b, opt_x)
    print("Optimal value is {}".format(f_opt))

    fig, ax = plt.subplots()

    # Plot the data using semilogy
    ax.semilogy(np.arange(len(history_proj)), (history_proj - f_opt) / f_opt, linewidth=2, label='Proejcted GD')
    ax.semilogy(np.arange(len(history_accel)), (history_accel - f_opt) / f_opt, linewidth=2, label='Fast Proejcted GD')
    ax.semilogy(np.arange(len(history_log)), (history_log - f_opt) / f_opt, linewidth=2, label='Mirror')
    ax.semilogy(np.arange(len(history_admm)), (history_admm - f_opt) / f_opt, linewidth=2, label='ADMM')
    # Set labels and title
    ax.set_xlabel('Number of iteration')
    ax.set_ylabel('Objective')
    ax.set_title('Objective v.s. Number of iteration')

    # Add a legend
    ax.legend()

    # Display the plot
    plt.show()
    # debug_project_simplex()