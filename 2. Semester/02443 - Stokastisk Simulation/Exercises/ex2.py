import numpy as np, pandas as pd, scipy.stats, sympy, matplotlib.pyplot as plt



p = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]


randNr = np.random.rand(10000)

def crude(U, p):
    edges = np.cumsum(p)
    out = (U < edges[0]) * 1
    for i in range(len(p) - 1):
        out = out + (np.logical_and(U > edges[i], U < edges[i + 1])) * (i + 2)
    return out


plt.figure()
nobs2, w, patch = plt.hist(crude(randNr, p), 6)


def rejection(U, p):
    out = []
    n = len(p)
    p_max = max(p)
    for i in range(len(U)):
        k = U[i]
        I = int(np.floor(n*k)) # slide 12/19
        U_2 = U[i-1]
        if U_2 < (p[I]/p_max):
            out.append(I + 1)
    return out

plt.figure()
nobs2, w, patch = plt.hist(rejection(randNr, p), 6)


def alias(U, p):
    eps = 1e-10

    n = len(p)
    L = list(range(1, n+1))

    F = n * np.asarray(p)
    #G = [j for j, k in enumerate( p ) if k <= 1]
    #S = [j for j, k in enumerate( p ) if k >= 1]
    G = np.where(F >= 1)[0]
    S = np.where(F <= 1)[0]


    # Slide 17/19
    while len(S)!=0:
        k = G[0]
        j = S[0]
        L[j] = k
        F[k] = F[k] - (1 - F[j])
        if F[k] < (1 - eps):
            G = np.delete(G,0)
            S = np.append(S, k)
        S = np.delete(S,0)
    out = []
    n = len(p)
    p_max = max(p)
    for i in range(len(U)):
        j = U[i]
        I = int(np.floor(n * j))  # slide 12/19
        U_2 = U[i - 1]
        # Slide 16/19. Genbruger kode fra reject, tilføjer bare et else, så vidt
        # jeg kan se.
        if U_2 < (F[I]):
            out.append(I + 1)
        else:
            out.append(L[I] + 1)
    return out

plt.figure()
nobs2, w, patch = plt.hist(alias(randNr, p), 6)
plt.show()