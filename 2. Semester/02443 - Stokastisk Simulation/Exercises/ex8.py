import numpy as np, pandas as pd, scipy.stats, sympy, matplotlib.pyplot as plt,matplotlib as mpl, math


def estimate_p(x, bootraps, a, b):
    """

    :param x: Input sample
    :param bootraps: The bootstrapped population
    :param a: lower value
    :param b: higher value
    :return: Returns the probability that the mean_bootstrap - true_mean is between these values
    """
    mean = np.mean(x)
    reps = len(bootraps)
    mean_bootraps = np.mean(bootraps, axis=1)
    count = np.sum([a < diff < b for diff in (mean_bootraps - mean)])
    p = count/reps
    return p


def bootstrap(x, r=10):
    """

    :param x: Input sample
    :param r: Defualt 10, defines how many bootstraps to run
    :return: Returns the bootstrapped estimate of the population input x
    """
    n = len(x)
    iters = [np.random.choice(x,n,replace=True) for _ in range(r)]
    return iters


def get_list_info(x):
    mean = np.mean(x)
    median = np.median(x)
    return mean, median


def get_bootstrap_info(bootstrap):
    mean_bootstrap_list = np.mean(bootstrap, axis=1)
    median_bootstrap_list = np.median(bootstrap, axis=1)
    mean_bootstrap = np.mean(mean_bootstrap_list)
    median_bootstrap = np.median(median_bootstrap_list)
    mean_var_bootstrap = np.var(mean_bootstrap_list)
    median_var_bootstrap = np.var(median_bootstrap_list)
    return mean_bootstrap, median_bootstrap, mean_var_bootstrap, \
            median_var_bootstrap, mean_bootstrap_list, median_bootstrap_list



# x = [56,101,78,67,93,87,64,72,80,69]

# b = bootstrap(x, r=1000)

# mean, median, var = get_list_info(x)

# print("Mean: {} -- Median: {} -- Var: {}".format(mean, median, var))

# meanbs, medianbs, varbs = get_bootstrap_info(b)

# print("Mean: {} -- Median: {} -- Var: {}".format(meanbs, medianbs, varbs))


def pareto(k, beta):
    """

    :param k:
    :param beta:
    :return: Returns the theoritical mean and variance
    """
    mean_p = k * beta/(k - 1)
    var_p = beta**(1/k)
    return mean_p, var_p