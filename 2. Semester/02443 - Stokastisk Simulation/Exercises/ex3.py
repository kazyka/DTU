import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.stats import norm, expon
import scipy
import time


def generate_exponential(lambd, length):
    # using notes by iversen page 55-56
    rand = np.random.rand(length, 1)
    T = -1 / lambd * np.log(rand)
    return T


def generate_pareto(beta, k, length):
    rand = np.random.rand(length, 1)
    out = beta * (np.power(rand, -1 / k) - 1)
    return out


def generate_normal(length, centralLimitnr=10, UseBoxMuller=True):
    if UseBoxMuller:
        rand = np.random.rand(length * 2, 1)
        rand1 = rand[:length]
        rand2 = rand[length:]
        z = np.sqrt(-2 * np.log(rand1)) * np.cos(2 * math.pi * rand2)
    else:
        # use central limit theorem
        rand = np.random.rand(length * centralLimitnr, 1)
        z = np.zeros((length, 1))
        for i in range(centralLimitnr):
            z = z + rand[length * i:length * (i + 1)]
        z = z - centralLimitnr / 2
    return z


if __name__ == "__main__":
    np.random.seed(123)
    size = 10000
    lambd = 1.5
    x = np.linspace(0, 7, 1000)
    myExp = lambd * np.exp(-lambd * x)

    # generate exponential data
    exp = generate_exponential(lambd, size)
    # myHist = plt.hist(exp,bins=100)
    # plt.show()
    # plt.plot(x,myExp)
    # plt.show()

    beta = 1
    k = [2.05, 2.5, 3, 4, 20, 200, 2000, 20000, 200000]
    xlim = [70, 30, 23, 9, 20, 20, 20, 20, 20]
    for i in range(len(k)):
        pareto = generate_pareto(beta, k[i], size)
        x = np.linspace(0, xlim[i], 1000)
        myPareto = np.power(1 + x / beta, -k[i])
        plt.plot(x,myPareto)
        plt.ylim(0,1)
        plt.show()
        plt.hist(pareto,bins=100)
        plt.show()
        parMean = beta * k[i] / (k[i] - 1)
        parVar = beta * beta * k[i] / ((k[i] - 1) * (k[i] - 1) * (k[i] - 2))
        print('pareto actual mean: {0:.7f} \t expected mean {1:.7f} \t k={2:.2f} \t mean ratio: {3:.7f}'.format(
            np.mean(pareto), parMean, k[i], np.mean(pareto) / parMean))
        print('pareto actual var: {0:.7f} \t expected var {1:.7f} \t k={2:.2f} \t var ratio: {3:.7f}'.format(
            np.var(pareto), parVar, k[i], np.var(pareto) / parVar))

    t = time.time()
    norm1 = generate_normal(size)
    print(time.time() - t)
    t = time.time()
    norm2 = generate_normal(size, centralLimitnr=10, UseBoxMuller=False)
    print(time.time() - t)
    t = time.time()
    norm3 = generate_normal(size, centralLimitnr=1000, UseBoxMuller=False)
    print(time.time() - t)
    t = time.time()
    norm4 = generate_normal(size, centralLimitnr=100000, UseBoxMuller=False)
    print(time.time() - t)

    # plt.hist(norm1,bins=100)
    # plt.show()
    # plt.hist(norm2,bins=100)
    # plt.show()
    # plt.hist(norm3,bins=100)
    # plt.show()
    # plt.hist(norm4,bins=100)
    # plt.show()





