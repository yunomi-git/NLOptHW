import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#################
# generating data

f = lambda x: x**2 / 3 - 3 * x * np.sin(x-2) - np.cos(x)
# f = lambda x: x*np.cos(x) / 3
m = lambda x: np.zeros_like(x)

#################

def K_rbf(x, y, k, sigma):
    # return k * np.exp(- np.linalg.norm(x - y, axis=-1) ** 2 / (2.0 * sigma))
    return k * np.exp(- 0.5 * ((x - y) / sigma)**2)

def pdf(x, mu, sigma):
    # print(1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - mu) / sigma)**2), x - mu, sigma)
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 0.5 * ((x - mu) / sigma)**2)

def repeat_first_dim(x, num_repeats):
    return np.stack(np.repeat(x[np.newaxis, :], axis=0, repeats=num_repeats), axis=0)

def compute_post(grid_x, m, observed_x, observed_f, k_rbf, sigma_rbf):
    # grid_x: the set of points we want to compute the posterior distribution
    # m: m(x) returns the prior guess
    # observed_x: observed points
    # observed_f: function value of observed points
    # k_rbf, sigma_rbf: parameters of RBF kernel

    # return mu_x, sigma_x: mean and variance of grid_x

    #############################
    # write your code here
    mu_x = np.zeros(len(grid_x))
    sigma_x = np.zeros(len(grid_x))
    num_observations = len(observed_x)

    # first construct cov matrix
    observations_stacked = repeat_first_dim(observed_x, num_observations)
    observations_transposed = np.swapaxes(observations_stacked, 0, 1)
    cov_matrix = K_rbf(observations_stacked, observations_transposed, k=k_rbf, sigma=sigma_rbf)

    # Then construct the error
    error = observed_f - m(observed_x)

    cov_error_product = np.linalg.solve(cov_matrix, error)

    # Calculate each value in grid
    for i in range(len(grid_x)):
        x = grid_x[i]
        input_cov = K_rbf(x, observed_x, k=k_rbf, sigma=sigma_rbf)

        mu_x[i] = m(x) + input_cov.T @ cov_error_product
        sigma_x[i] = K_rbf(x, x, k=k_rbf, sigma=sigma_rbf) - input_cov.T @ np.linalg.solve(cov_matrix, input_cov)

    sigma_x[sigma_x < 0] = 1e-10
    ##############################

    return mu_x, sigma_x


def EI(f_star, mu_x, sigma_x, grid_x):
    # f_star: best value observed so far
    # grid_x: the set of points we want to compute the expected improvement
    # mu_x, sigma_x: mean and variance of grid_x

    # return EI_x: expected improvement at grid_x

    #############################
    # write your code here
    EI_x = np.zeros(len(grid_x))
    for i in range(len(grid_x)):
        sigma = sigma_x[i]
        mu = mu_x[i]
        # EI_x[i] = (sigma ** 2 * pdf(f_star, mu, sigma) +
        #            (f_star - mu) * norm.cdf((f_star - mu / sigma)))
        EI_x[i] = (sigma ** 2 * norm.pdf(f_star, scale=sigma, loc=mu) +
                   (f_star - mu) * norm.cdf(f_star, loc=mu, scale=sigma))

    ##############################
    return EI_x


def LCB(f_star, mu_x, sigma_x, grid_x, beta):
    # f_star: best value observed so far
    # grid_x: the set of points we want to compute the lower confidence bound
    # mu_x, sigma_x: mean and variance of grid_x

    # return LCB_x: (negative) lower confidence bound at grid_x

    #############################
    # write your code here
    LCB_x = mu_x - beta * sigma_x

    ##############################

    return LCB_x


def Bayesian_EI(max_iter=20, k_rbf=1, sigma_rbf=1, grid_range=[-10, 10], grid_freq=0.1):
    # max_iter: maximum number of iteration
    # k_rbf, sigma_rbf: parameters of RBF kernel
    # grid_range, grid_freq: setting the grid

    # grid of x
    grid_x = np.arange(grid_range[0], grid_range[1], grid_freq)

    # initialize observed set
    observed_x = np.array([grid_x[int(len(grid_x) / 2)]])

    # record which points have been queried
    unqueried_index = np.full(grid_x.shape, True)
    unqueried_index[int(len(grid_x) / 2)] = False

    # record mu and sigma at each iteration
    mu_trajectory = np.zeros((len(grid_x), max_iter))
    sigma_trajectory = np.zeros((len(grid_x), max_iter))

    observed_f = f(observed_x[0])
    f_star = observed_f

    for i_iter in range(max_iter):
        #############################
        # write your code here

        # Calculate the gaussian process model over all points
        mu_x, sigma_x = compute_post(grid_x=grid_x,
                                     observed_x=observed_x,
                                     observed_f=observed_f,
                                     m=m,
                                     k_rbf=k_rbf,
                                     sigma_rbf=sigma_rbf)

        # Calculate acquisition for all points
        acquisition_grid = EI(f_star,
                              mu_x=mu_x[unqueried_index],
                              sigma_x=sigma_x[unqueried_index],
                              grid_x=grid_x[unqueried_index])

        # Query point is the maximization
        descending_order = np.argsort(acquisition_grid)[::-1]
        grid_indices = np.arange(len(grid_x))[unqueried_index]
        grid_indices = grid_indices[descending_order]
        index_to_query = grid_indices[0]
        # index_to_query = np.argmax(acquisition_grid[unqueried_index])
        x_query = grid_x[index_to_query]
        f_query = f(x_query)

        # Save values
        if f_query < f_star:
            f_star = f_query

        observed_x = np.append(observed_x, x_query)
        observed_f = np.append(observed_f, f_query)
        mu_trajectory[:, i_iter] = mu_x
        sigma_trajectory[:, i_iter] = sigma_x
        unqueried_index[index_to_query] = False

        ##############################

    return observed_x, mu_trajectory, sigma_trajectory


def Bayesian_LCB(max_iter=20, k_rbf=1, sigma_rbf=1, beta=1, grid_range=[-10, 10], grid_freq=0.1):
    # max_iter: maximum number of iteration
    # k_rbf, sigma_rbf: parameters of RBF kernel
    # beta: beta in LCB
    # grid_range, grid_freq: setting the grid

    # grid of x
    grid_x = np.arange(grid_range[0], grid_range[1], grid_freq)

    # initialize observed set
    observed_x = np.array([grid_x[int(len(grid_x) / 2)]])

    # record which points have been queried
    unqueried_index = np.full(grid_x.shape, True)
    unqueried_index[int(len(grid_x) / 2)] = False

    # record mu and sigma at each iteration
    mu_trajectory = np.zeros((len(grid_x), max_iter))
    sigma_trajectory = np.zeros((len(grid_x), max_iter))

    observed_f = f(observed_x[0])
    f_star = observed_f

    for i_iter in range(max_iter):
        #############################
        # write your code here
       # Calculate the gaussian process model over all points
        mu_x, sigma_x = compute_post(grid_x=grid_x,
                                     observed_x=observed_x,
                                     observed_f=observed_f,
                                     m=m,
                                     k_rbf=k_rbf,
                                     sigma_rbf=sigma_rbf)

        # Calculate acquisition for all points
        acquisition_grid = LCB(f_star,
                               mu_x=mu_x[unqueried_index],
                               sigma_x=sigma_x[unqueried_index],
                               grid_x=grid_x[unqueried_index],
                               beta=beta)

        # Query point is the maximization
        ascending_order = np.argsort(acquisition_grid)
        grid_indices = np.arange(len(grid_x))[unqueried_index]
        grid_indices = grid_indices[ascending_order]
        index_to_query = grid_indices[0]
        # index_to_query = np.argmax(acquisition_grid[unqueried_index])
        x_query = grid_x[index_to_query]
        f_query = f(x_query)

        # Save values
        if f_query < f_star:
            f_star = f_query
        observed_x = np.append(observed_x, x_query)
        observed_f = np.append(observed_f, f_query)
        mu_trajectory[:, i_iter] = mu_x
        sigma_trajectory[:, i_iter] = sigma_x
        unqueried_index[index_to_query] = False
        ##############################

    return observed_x, mu_trajectory, sigma_trajectory

if __name__=="__main__":
    # plot (try different sigma and beta)
    grid_range = [-10, 10]
    num_runs = 20
    observed_x, mu_trajectory, sigma_trajectory = Bayesian_EI(max_iter = num_runs, k_rbf=1, sigma_rbf=5, grid_range=grid_range)
    # observed_x, mu_trajectory, sigma_trajectory = Bayesian_LCB(max_iter = num_runs, k_rbf=1, sigma_rbf=5, beta = 1, grid_range=grid_range)

    # Generate x values
    grid_x = np.arange(grid_range[0], grid_range[1], 0.1)

    # Create the figure and axes objects
    fig, axes = plt.subplots(nrows=num_runs//2, ncols=2, figsize=(10, 40))

    for i_ax in range(len(axes.flatten())):
        ax = axes.flatten()[i_ax]
        # Plot the mean
        ax.plot(grid_x, mu_trajectory[:, i_ax], 'r', label='mean $\mu(x)$')

        # Fill between mu +/- sigma
        ax.fill_between(grid_x, mu_trajectory[:, i_ax] - sigma_trajectory[:, i_ax],
                        mu_trajectory[:, i_ax] + sigma_trajectory[:, i_ax],
                        color='gray', alpha=0.5, label='deviation $\pm \sigma(x)$')

        ax.scatter(observed_x[:i_ax], f(observed_x[:i_ax]))
        ax.scatter(observed_x[i_ax:i_ax + 1], f(observed_x[i_ax:i_ax + 1]), color='y', label='query points')

        ax.plot(grid_x, f(grid_x), 'g', label="f(x)")

        # Setting labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('t=' + str(i_ax + 1))
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()