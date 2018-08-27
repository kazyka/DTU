import numpy as np


def crudeMC(dist):
    out = np.mean(dist)
    return out

def antitheticMC(uniform):
    exp1 = np.exp(uniform)
    exp2 = np.exp(1-uniform)
    Y = (exp1+exp2)/2
    out = np.mean(Y)
    return out

def controlVarMC(uniform):
    X = np.exp(uniform)
    mu = np.mean(uniform)
    c = -np.cov(X,uniform)[0,1]/np.var(uniform)
    Z = X+c*(uniform-mu)
    out = np.mean(Y)
    return out


if __name__ == '__main__':
    nrSims = 100
    simSize = 100
    uniform = np.random.rand(simSize)
    expDist = np.exp(uniform)
    crudeMeanExp = []
    for i in range(nrSims):
        out = crudeMC(expDist[i*simSize:],simSize)
        crudeMeanExp.append(out)
    mean = np.mean(crudeMeanExp)
    var = np.var(crudeMeanExp)
    upperC = mean+np.sqrt(var)/np.sqrt(nrSims)*1.96
    lowerC = mean-np.sqrt(var)/np.sqrt(nrSims)*1.96
    print('mean: '+ str(mean))
    print('upper: '+ str(upperC))
    print('lower: '+ str(lowerC))

    ##Antithetic
    expDist2 = np.exp(1-uniform)
    Y = (expDist+expDist2)/2
