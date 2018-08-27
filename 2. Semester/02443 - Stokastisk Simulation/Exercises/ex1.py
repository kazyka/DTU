import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import stats
from scipy.misc import factorial


def LCG(x0, a, c, M, nrSamples):
    out = []
    rand = x0
    for i in range(nrSamples):
        rand = (a * rand + c) % M
        out.append(rand / M)
    return out


def countBins(observations, binEdges):
    out = [0] * (len(binEdges) + 1)
    extendedEdges = binEdges.insert(0, -9999999999999999999999)
    extendedEdges = binEdges.append(9999999999999999999999)
    for obs in observations:
        for i in range(len(out)):
            if (obs > binEdges[i] and obs < binEdges[i + 1]):
                out[i] += 1
    return out


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
        #nObserved = list(plt.hist(LCG_rand, bins=10)[0])
        nObserved, _, _ = plt.hist(obs, bins=nrBins)
    else:
        #binEdges = list(np.linspace(min, max, nrBins + 1)[1:-1])
        #nObserved = countBins(obs, binEdges)
        nObserved, _, _ = plt.hist(obs, bins=nrBins)
        #print(nObserved)
        #n, bins, patches = plt.hist(obs, bins=nrBins)
        #print("n={}, bins={} and patches={}".format(n,bins, patches))

    for i in range(len(nObserved)):
        T += math.pow((nObserved[i] - nExpected[i]), 2) / nExpected[i]
    return T




def chi_square(obsDist, expectDist):
    T = 0
    for i in range(len(obsDist)):
        T += math.pow((obsDist[i] - expectDist[i]), 2) / expectDist[i]
    return T



def hypo_test_KS(D, n, allKnown=True):
    if allKnown:
        res = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D

    if res < 1.358:
        return "Accepted"
    else:
        return "Rejected"


def kolomogrov_Smirnov(obs, nrBins, dist='uniform', usePlt=False):
    if dist == 'uniform':
        nExpected = np.asarray([len(obs) / nrBins] * nrBins) / len(obs)

    if usePlt:
        nObserved = list(plt.hist(LCG_rand, bins=10)[0])
    else:
        #binEdges = list(np.linspace(min, max, nrBins + 1)[1:-1])
        #nObserved = countBins(obs, binEdges)
        nObserved, _, _ = plt.hist(obs, bins=nrBins)
    nObserved = np.asarray(nObserved) / len(obs)
    Fn = np.cumsum(nObserved)
    F = np.cumsum(nExpected)

    D = np.max(np.abs(Fn - F), axis=0)

    return D


def runTest_above_below(obs, nrbins=10):
    plt.clf()
    bins = plt.hist(obs, bins=nrbins)[0]

    n1 = np.sum((bins) > 1000)
    n2 = len(bins) - n1

    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (math.pow((n1 + n2), 2) * (n1 + n2 - 1))

    diffBin = bins[:-1] - bins[1:]
    changeDown = np.logical_and(diffBin[:-1] > 0, diffBin[1:] < 0)
    changeUp = np.logical_and(diffBin[:-1] < 0, diffBin[1:] > 0)
    change = np.logical_or(changeDown, changeUp)
    N = np.sum(change)

    conf = stats.norm(mu, var).ppf(0.975)

    if N < conf:
        return "Accepted"
    else:
        return "Rejected"


def runTest_up_down(Uvec):
    Uvec = np.asarray(Uvec)
    diffU = Uvec[:-1] - Uvec[1:]
    changeDown = np.logical_and(diffU[:-1] > 0, diffU[1:] < 0)
    changeUp = np.logical_and(diffU[:-1] < 0, diffU[1:] > 0)
    change = np.logical_or(changeDown, changeUp)
    changeInds = np.where(change)[0] + 1

    Ulengths = (changeInds[1:] - changeInds[:-1])
    Ulengths = np.insert(Ulengths, 0, changeInds[0], axis=0)

    n = len(Uvec)
    R_i_n = np.empty((10, 1))
    our_runs = np.empty((10, 1))
    i = 1
    while i < 11:
        R_i_n[i - 1] = (2 * (i * i + 3 * i + 1) * n - (i * i * i + 3 * i * i - i - 4)) / factorial(i + 3)
        our_runs[i - 1] = np.sum(Ulengths == i)
        i += 1

    Tval = chi_square(our_runs, R_i_n)
    res = hypo_test_chi(Tval, 10, 0.975)

    return res


def correlation_test(Uvec, h):
    c = np.empty((h, 1))
    n = len(Uvec)
    hCount = 1
    while hCount <= h:
        for iCount in range(n - h):
            c[hCount - 1] += Uvec[iCount] * Uvec[iCount + hCount]
        c[hCount - 1] = c[hCount - 1] / (n - hCount)
        hCount += 1

    # use KS test to verify
    nrBins = 100
    plt.clf()
    hist = plt.hist(c, nrBins, range=(0, 1))
    propabilities = hist[0] / h
    binEdges = hist[1]
    dist = stats.norm(0.25, 7 / (144 * n))
    theory = np.empty((nrBins, 1))
    for i in range(nrBins):
        theory[i] = dist.cdf(binEdges[i + 1]) - dist.cdf(binEdges[i])
    D = np.max(theory - propabilities)
    out = hypo_test_KS(D, n)
    return out


if __name__ == '__main__':
    plt.clf()
    M = math.pow(2, 32)
    a = 1664525
    c = 1013904223
    LCG_rand = LCG(3, a, c, M, 10000)
    M = math.pow(2,4)
    a = 5
    c = 1
    LCG_rand2 = LCG(3, a, c, M, 10000)

    plt.figure()
    plt.hist(LCG_rand, bins=10)
    plt.title("With $x_0 = 3$ $a=1664525$, $c=1013904223$ and $M=2^{32}$")
    plt.savefig("LCGperfect.eps", bbox_inches="tight")


    plt.figure()
    plt.scatter(LCG_rand[:-1], LCG_rand[1:], 0.1)
    plt.savefig("LCGperfectscatter.eps", bbox_inches="tight")

    plt.figure()
    plt.hist(LCG_rand2, bins=10)
    plt.title("With $x_0 = 3$ $a=5, $c=1 and $M=2^{4}$")
    plt.savefig("LCGslide.eps", bbox_inches="tight")

    plt.figure()
    plt.scatter(LCG_rand2[:-1], LCG_rand2[1:], 0.1)
    plt.savefig("LCGslidescatter.eps", bbox_inches="tight")
    #plt.show()

    T = chi_square_distribution(LCG_rand, 10, usePlt=False)
    print("Chi2 test: " + hypo_test_chi(T, 9, 0.975))

    D = kolomogrov_Smirnov(LCG_rand, 10, usePlt=True)
    print("KS test: " + hypo_test_KS(D, 10))

    run1 = runTest_above_below(LCG_rand)
    print('run test above below: ' + run1)

    run2 = runTest_up_down(LCG_rand)
    print('run test up down: ' + run2)

"""
    # try with np random numbers
    print("\n\nWith np random numbers")
    np.random.seed(123)
    rand = np.random.rand(10000, 1)

    T = chi_square_distribution(rand, 10, 0, 1, usePlt=True)
    print("Chi2 test: " + hypo_test_chi(T, 9, 0.975))

    D = kolomogrov_Smirnov(rand, 10, 0, 1, usePlt=True)
    print("KS test: " + hypo_test_KS(D, 10))

    run1 = runTest_above_below(rand)
    print('run test above below: ' + run1)

    run2 = runTest_up_down(rand)
    print('run test up down: ' + run2)
    """


