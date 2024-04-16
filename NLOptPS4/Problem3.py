import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

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

def bisection_search(func, lower_x=-1.0, upper_x=1.0, tolerance=1e-8, max_steps = 100):
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


def enforce_positive(x):
    x[x<0] = 0
    return x

def get_gradient(A, b, x):
    return 2 * A.T @ (A @ x - b)

def get_objective(A, b, x):
    return 0.5 * np.square(np.linalg.norm(A @ x - b))

def project_onto_simplex(x):
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

def project_simplex_xlogx(x, learning_rate, gradient):
    proj = x * np.exp(learning_rate * gradient)
    proj /= np.sum(proj)
    return proj

def optimize_admm(x, learning_rate, gradient):
    # optimizer f(x)
    x = optimize(f, alpha, y)
    y = optimize(g, alpha, x)
    alpha = x + y / 2
    

def optimize_cvxpy(A, b, n):
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.square(cp.norm(A @ x - b)))
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    constraints = [x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(obj,
                      constraints)
    prob.solve()
    x_sol = np.array(x.value)
    # x_sol[x_sol < 1e-4] = 0 # TODO remove
    return x_sol
    # print(x_sol)

def optimize_gradient_descent(A, b, x0, steps, learning_rate, proj_method="euclid"):
    # methods: euclid, log
    x_history = [x0]
    x = x0
    for i in range(steps):
        gradient = get_gradient(A, b, x)
        x = x - learning_rate * gradient
        # Project
        if proj_method == "euclid":
            x = project_onto_simplex(x)
        else:
            x = project_simplex_xlogx(x, learning_rate, gradient)
        x_history.append(x)
    # print(x)
    return np.array(x_history)

def optimize_nesterov_accelerated(A, b, x0, steps, learning_rate):
    x_history = [x0]
    x_1 = x0
    x = x0
    for t in range(steps):
        y = x + (t-2) / (t+1) * (x - x_1)
        gradient = get_gradient(A, b, y)
        y = y - learning_rate * gradient
        # Project
        y = project_onto_simplex(y)
        # Save
        x_1 = x
        x = y
        x_history.append(x)
    # print(x)
    return np.array(x_history)

def optimize_ADMM(A, b, x0, steps, learning_rate):
    # methods: euclid, log
    x_history = [x0]
    x = x0
    for i in range(steps):
        gradient = get_gradient(A, b, x)
        x = x - learning_rate * gradient
        # Project
        x = project_onto_simplex(x)
        x_history.append(x)
    # print(x)
    return np.array(x_history)

def plot_distance_to_objective(ax, x_history, x_optimal, A, b, name):
    f_history = []
    for i in range(len(x_history)):
        f_history.append(get_objective(A, b, x_history[i]))
    f_history = np.array(f_history)
    # f_history = get_objective(A, b, x_history)
    f_opt = get_objective(A, b, x_optimal)

    line = ax.plot(f_history - f_opt, label=name)
    # line.set_label(name)

def debug_project_simplex():
    point = np.array([200, 10, 10])
    x = project_onto_simplex(point)
    x_log = project_simplex_xlogx(point, 0.1, 1/point)
    print(x)
    print(x_log)

if __name__=="__main__":
    np.set_printoptions(precision=5)

    m = 500
    n = 200
    A = np.random.rand(m, n)
    b = np.random.rand(m)
    # x0 = np.random.rand(n)
    x0 = np.zeros(n)

    history_accel = optimize_nesterov_accelerated(A, b, x0, steps=500, learning_rate=0.0001)

    history_proj = optimize_gradient_descent(A, b, x0, steps=500, learning_rate=0.0001, proj_method="euclid")
    history_log = optimize_gradient_descent(A, b, x0, steps=500, learning_rate=0.0001, proj_method="log")

    x_optimal = optimize_cvxpy(A, b, n)
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    plot_distance_to_objective(ax, history_accel, x_optimal, A, b, "nesterov acceleration")
    plot_distance_to_objective(ax, history_proj, x_optimal, A, b, "projected euclidian")
    plot_distance_to_objective(ax, history_log, x_optimal, A, b, "projected log")
    ax.legend()
    plt.show()
    # debug_project_simplex()