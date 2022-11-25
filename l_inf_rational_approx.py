#######################################################################################################################
# Conversion to python of RationalMinMaxOpt.m from https://github.com/nirsharon/rational_approx, that implements the
# manuscript: "Rational approximation and its application to improving deep learning classifiers", by
# V. Peiris, N. Sharon, N. Sukhorukova J. Ugon. See (arXiv link) https://arxiv.org/abs/2002.11330.
# Written by David Denisov.
#######################################################################################################################
import numpy as np
from scipy.optimize import linprog
from algorithms import predict, rg
import matplotlib.pyplot as plt
from time import perf_counter as time
from algorithms import predict, rg, robust_mean
from algorithms import Fast_centroid_set, find_arg_min
from scipy.optimize import minimize
import json

import warnings

warnings.filterwarnings("ignore")


def chebeval_scalars(coef, pts, m):
    return np.polynomial.chebyshev.chebval(pts, coef)


def chebeval_var(pts, n):
    I = np.eye(n)
    I[0, 0] = 2
    Tn = np.zeros(len(pts), n)
    for deg in range(n):
        Tn[:, deg + 1] = chebeval_scalars(I[deg + 1, :], pts, deg + 1)
    return Tn


def barycentric_poly_inter2(xj, yj, x):
    n = len(xj)
    thd = 1e-10

    W = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            W[j, k] = 1 / (xj[j] - xj[k])

    W[np.eye(n) == 1] = 1
    omegaj = np.prod(W, 0)

    # the matrix 1/(x-x_j)
    I = np.eye(n)
    A = 1 / (x - xj)

    # too close
    if np.min(np.abs(x - xj)) < thd:
        ind = np.argwhere(np.abs(x - xj) < thd)
        row, col = ind[:, 0], ind[:, 1]
        for i in range(len(row)):
            A[row[i], :] = I[col[i], :]

    B = A * np.diag(omegaj)
    w = B / sum(B, 2)

    return w * yj


# The Matlab multRow, translated to python as is, note that it seems that it can be done using matrix operations.
def multRow_slow(F, Tm):
    M = np.zeros_like(Tm)
    for i in range(len(F)):
        M[i, :] = F[i] * Tm[i, :]
    return M


# The Matlab multRow optimized such that it will use matrix operations and not the for as above.
# Uncomment the commented code to debug this function (if the code is correct this should print 0 up rounding error).
def multRow(F, Tm):
    # M1 = multRow_slow(F, Tm)
    M2 = F.reshape((-1, 1)).repeat(len(Tm[0]), 1) * Tm
    # print(np.abs(M1-M2).sum())
    return M2


def LpRat(z, k, X, Y, Tn, Tm, DLB, DUB, r_l=0):
    cond1 = multRow(Y + z, Tm)
    cond2 = multRow(Y - z, Tm)
    Z, I = np.zeros_like(Tn), np.ones([len(X), 1])
    A1 = np.concatenate([Tn, -cond1, -I], 1)
    A2 = np.concatenate([-Tn, cond2, -I], 1)
    A3 = np.concatenate([Z, -Tm, np.zeros([len(X), 1])], 1)
    A4 = np.concatenate([Z, Tm, np.zeros([len(X), 1])], 1)

    A = np.concatenate([A1, A2, A3, A4], 0)

    b = np.concatenate([np.zeros(2 * len(X)), -DLB * np.ones(len(X)), DUB * np.ones(len(X))], 0)

    lb = -np.inf * np.ones(2 * k + 2)
    lb[2 * k + 1] = -1
    lb[k] = 0.1

    ub = np.inf * np.ones(2 * k + 2)
    ub[2 * k + 1] = 1

    obj = np.zeros(2 * k + 2)
    obj[2 * k + 1] = 1

    L = list(zip(lb, ub))

    # d = {'tol':1e-12}
    ans = linprog(obj, A, b, bounds=L, method='Highs')  # ,'disp':True
    if ans.success:
        x = ans.x

        p = x[:k]
        q = x[k:2 * k + 1]
        u = x[2 * k + 1]
        return p, q, u
    else:
        return None, None, 3
        #   if r_l >= 3:
        #       return None, None, 3
        #   else:
        #       return LpRat(z, k, X, Y + rg.normal(0, 1e-3, len(Y)), Tn, Tm, DLB, DUB, r_l+1)


def checkVal(z, k, X, Y, Tn, Tm, DLB, DUB):
    u = LpRat(z, k, X, Y, Tn, Tm, DLB, DUB)[2]
    return u <= 1e-15


# Since we use this for comparison for l_1 loss there is no reason to find the "true" l_inf minimum or approximate
# the value too much, and up to 1e-3 (0.001) seems sufficient.
def rat_min_max(X, Y, k, uH=None, lim=1e-3, debug=False):
    Tn = np.polynomial.polynomial.polyvander(X, k - 1)
    Tm = np.polynomial.polynomial.polyvander(X, k)

    ans, uL = [], 0

    if uH is None:
        f = np.polyfit(X, Y, k)
        uH = np.abs(np.polyval(np.flipud(f), X) - Y).max()

        ans = [f, np.ones(1)]

    while (uH - uL) > lim:
        z = (uH + uL) / 2
        # If the optimal result is lower than distance z
        cur = LpRat(z, k, X, Y, Tn, Tm, 1e-12, 1e12)
        if len(ans) > 0 and debug:
            print(uH, uL, np.abs(pol(X, ans[0]) / pol(X, ans[1]) - Y).max())
        if cur[2] < lim/4:
            ans = [cur[0], cur[1]]
            uH = z
        else:
            uL = z

    # Calculates the optimal p, q
    return ans


def r_inf(X, Y, k, uH=None, lim=1e-3, debug=False):
    a, b = rat_min_max(X, Y, k, uH, lim, debug)
    b1 = b[0]
    return a / b1, b[1:] / b1


def pol(X, c):
    return np.polyval(np.flip(c), X)


def sci_loss(x, *args):
    X, Y, m = args
    c, d = x[:m], x[m:]
    return np.abs(predict(X, [c, d]) - Y).sum()


def test():
    n, k = pow(10, 4), 1
    X, q = np.linspace(0, 10, n), np.ones((2, k))
    Y = predict(X, q) + rg.normal(0, 0.01, n)

    t = time()
    options = Fast_centroid_set(X, Y, 128, k)
    func = find_arg_min(X, Y, options).reshape(2 * k)
    x = minimize(sci_loss, func, (X, Y, k)).x
    t1 = time() - t
    diff1 = np.abs(predict(X, [x[:k], x[k:]]) - Y)
    m1, m1s = diff1.max(), diff1.mean()

    t = time()
    a, b = r_inf(X, Y, k, m1)
    t2 = time() - t

    diff = np.abs(predict(X, [a, b]) - Y)
    m2, m2s = diff.max(), diff.mean()

    print(m2 / m1, m2s / m1s, t2 / t1)


if __name__ == '__main__':
    for _ in range(10):
        test()
