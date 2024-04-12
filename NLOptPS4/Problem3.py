import numpy as np

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

def bisection_search(func, lower_x=-1.0, upper_x=1.0, tolerance=1e-8):
    lower_x, upper_x = find_bounds(func, lower_x, upper_x)
    # for decreasing functions
    test_x = (lower_x + upper_x) / 2.0
    test_val = func(test_x)
    while np.abs(test_val) > tolerance:
        if test_val > 0:
            lower_x = test_x
        else:
            upper_x = test_x
        test_x = (lower_x + upper_x) / 2.0
        test_val = func(test_x)
    return test_x


def enforce_positive(x):
    x[x<0] = 0
    return x

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

if __name__=="__main__":
    point = np.array([2, 2, 2])
    x = project_onto_simplex(point)
    print(x)