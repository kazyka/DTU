import numpy as np, pandas as pd, scipy.stats, sympy
import matplotlib.pyplot as plt,matplotlib as mpl, math
from math import factorial
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import stats
from scipy.misc import factorial
from matplotlib import rc


def generate_normal(length, centralLimitnr=10, UseBoxMuller=True):
    if UseBoxMuller:
        rand = np.random.rand(length * 2, 1)
        rand1 = rand[:length]
        rand2 = rand[length:]
        z1 = np.sqrt(-2 * np.log(rand1)) * np.cos(2 * math.pi * rand2)
        return z1
    else:
        # use central limit theorem
        rand = np.random.rand(length * centralLimitnr, 1)
        z = np.zeros((length, 1))
        for i in range(centralLimitnr):
            z = z + rand[length * i:length * (i + 1)]
        z = z - centralLimitnr / 2
    return z


z = generate_normal(10000, centralLimitnr=10000, UseBoxMuller=True)

def hypo_test_chi(tVal, df, quantile):
    hypo = stats.chi2.ppf(quantile, df)
    if tVal >= hypo:
        return "Rejected"
    else:
        return "Accepted"


def chi_square_distribution(obs, nrBins, dist='uniform', usePlt=False):
    T = 0
    if dist == 'uniform':
        nExpected = [len(obs) / nrBins] * nrBins
    if usePlt:
        nObserved, _, _ = plt.hist(obs, bins=nrBins)
    else:
        nObserved, _, _ = plt.hist(obs, bins=nrBins)
    for i in range(len(nObserved)):
        T += math.pow((nObserved[i] - nExpected[i]), 2) / nExpected[i]
    return T


T = chi_square_distribution(z, 10, usePlt=False)
print(T)
print("Chi2 test: " + hypo_test_chi(T, 9, 0.975))
# print("\nT value is {}".format(T))
plt.close()




def conf_plot(Z, bins, r, N = 10000, plottingT = False, plottingChi = False):
    Tarr = []
    nobsArr = []
    Uarr = []
    b = np.linspace(r[0], r[1], num=bins)
    weight = np.array([scipy.stats.norm.cdf(j) - scipy.stats.norm.cdf(b[i]) for i, j in enumerate(b[1:])])
    for i in range(100):
        np.random.seed(i)
        U = Z
        Uarr.append(U)
        nobs, _ = np.histogram(U, bins=bins-1, range=r)
        nobsArr.append(nobs)
        chi_difference = ((nobs - N * weight) ** 2 / (N * weight)).sum()
        Tarr.append(chi_difference)
    T = np.mean(Tarr)
    if plottingT:
        plt.plot(Tarr, label="T-values")
        plt.plot([0, 100], [T, T], '--', c='gray', label="Mean")
        plt.xlabel("Test \#")
        plt.ylabel("T")
        plt.legend()
        plt.show()
    if plottingChi:
        n = 8
        scale = 4
        plt.plot([i / scale for i in range(20 * scale)],
                 [scipy.stats.chi2.pdf(i / scale, df=n - 1) for i in range(20 * scale)], label="$\chi^2 $")
        plt.plot([T, T], [-1, 1], '--g', label="$T_{mean}$")
        plt.plot([scipy.stats.chi2.ppf(0.975, df=9), scipy.stats.chi2.ppf(0.975, df=n - 1)], [-1, 1], '--r',
                 label="$95\%$ CI")
        plt.plot([scipy.stats.chi2.ppf(0.025, df=9), scipy.stats.chi2.ppf(0.025, df=n - 1)], [-1, 1], '--r')
        plt.ylim([0, 0.15])
        plt.ylabel("Propability")
        plt.xlabel("Value")
        plt.legend()
        plt.show()

