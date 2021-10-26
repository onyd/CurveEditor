from math import sqrt
import numpy as np


def factorise(linf, ldiag):
    t1 = sqrt(ldiag[0])
    L_linf, L_ldiag = [linf[0] / t1], [t1]
    for j in range(1, len(linf)):
        L_ldiag.append(sqrt(ldiag[j] - L_linf[-1]**2))
        L_linf.append(linf[j] / L_ldiag[-1])

    L_ldiag.append(sqrt(ldiag[-1] - L_linf[-1]**2))

    return L_linf, L_ldiag


def descente(linf, ldiag, b):
    Y = [b[0] / ldiag[0]]
    for i in range(1, len(ldiag)):
        Y.append((b[i] - linf[i - 1] * Y[i - 1]) / ldiag[i])

    return Y


def remonte(linf, ldiag, y):
    U = [y[-1] / ldiag[-1]]
    for i in range(len(ldiag) - 2, -1, -1):
        U.append((y[i] - linf[i] * U[len(ldiag) - i - 2]) / ldiag[i])

    return list(reversed(U))


def bidiag(a, b, k1=-1, k2=0):
    return np.diag(a, k1) + np.diag(b, k2)


def product(linf, ldiag):
    A = bidiag(linf, ldiag)
    return A.dot(A.transpose())


def solve(linf, ldiag, b):
    L_linf, L_ldiag = factorise(linf, ldiag)

    Y = descente(L_linf, L_ldiag, b)
    X = remonte(L_linf, L_ldiag, Y)

    return X
