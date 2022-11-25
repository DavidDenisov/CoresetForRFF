from time import perf_counter as time
from algorithms import predict, streaming_coreset, simplified_streaming_coreset, rg, coreset_sample, Bi_pred
from scipy.optimize import minimize
from Compute_sen import compute_sen_rat
import numpy as np
from l_inf_rational_approx import r_inf


def sci_loss(q, *args):
    X, Y, k = args
    c, d = q[:k], q[k:]
    return np.abs(predict(X, [c, d]) - Y).sum()


def compute_options(X, Y, k, runs, reuses):
    T, approx_arr = np.zeros((6, reuses, runs)), []
    beta = pow(2, k + 3)

    t = time()
    A = streaming_coreset(X, Y + rg.normal(0, 1e-5, len(Y)), k, beta=beta, reduce_beta=beta, spesific=True)
    T[0, :, :] = time() - t
    approx_arr.append(A)
    b = len(A[1])

    t = time()
    func = simplified_streaming_coreset(X, Y + rg.normal(0, 1e-5, len(Y)), k, beta=beta, reduce_beta=beta)[2]
    T[1, :, :] = time() - t
    approx_arr.append(func)

    t = time()
    f = minimize(sci_loss, np.zeros(2 * k), (X, Y, k)).x
    f = f.reshape((2, k))
    T[2, :, :] = time() - t
    approx_arr.append(f)
    while True:
        try:
            Y_ = Y + rg.normal(0, 1e-5, len(Y))
            l_max_upper = np.abs(predict(X, func) - Y_).max() + 1e-4
            t = time()
            f = r_inf(X, Y_, k, l_max_upper, lim=1e-4)
            T[3, :, :] = time() - t
            approx_arr.append(f)
            break
        except ValueError:
            print('(k,n) =', (k, len(Y)), ' l_inf-min failed, will add a resampled noise and re-try.')

    Y_, sen = Y, None
    while True:
        try:
            t = time()
            P = np.concatenate([X.reshape(-1, 1), Y_.reshape(-1, 1)], 1)
            sen = compute_sen_rat(P, k)
            T[5, :, :] = time() - t
            break
        except np.linalg.LinAlgError:
            Y_ = Y + rg.normal(0, 1e-5, len(Y))  # added to hopefully fix the problems that caused the throw.

    return approx_arr, sen, T, b


def core_sample1(X, Y, F, CoreSetSize, X_=None):
    if X_ is None:
        P_pred = predict(X, F)
        core_size = np.size(F)
    else:
        P_pred = Bi_pred(X_, F)
        core_size = np.size(F) + len(F)

    # Reshape Y
    miss = np.abs(P_pred - Y)
    sample, C, D, W = coreset_sample(X, Y, CoreSetSize, miss, P_pred)
    # For each sample we save x value (or index), original y-value, and wight (the projected y-value can be derived
    # from the known x value and the projections, i.e. the y-values are Bi_pred(X_, F)[sample]).
    core_size += np.size(sample) * 3

    return [P_pred, sample, C, D, W], core_size + 3  # representation of X


def core_sample2(X, Y, s, CoreSetSize):
    p = s / s.sum()
    n = len(X)

    samp = rg.choice(n, CoreSetSize, True, p=p)

    # Compute the frequencies of each sampled item; inspired by ...
    hist = np.histogram(samp, bins=range(n))[0].flatten()
    indxs = np.nonzero(hist)[0]
    w = 1 / (CoreSetSize * p)
    w = w[indxs] * hist[indxs]
    x, y = X[indxs], Y[indxs]

    # Ignoring representation of the weighs, just in case that for the uniform sample the weighs are np.ones_like(X)/n.
    return [x, y, w], np.size(w) * 2
