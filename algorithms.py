#####################################################################################################
#     Paper: PTSA for HMMs using coresets                                                           #
#     Implemented by David Denisov. daviddenisov14@gmail.com .                                      #
#     Code for the algorithms that are connected to the rational part of the paper // The           #
#     reduce algorithm based on the analysis mentioned in the paper is in 'reduce_full_algorithm'.  #
#####################################################################################################

from numpy.random import Generator, PCG64
from itertools import combinations, product
from scipy.special import comb
from scipy.optimize import minimize
from joblib import Parallel, delayed
from Unified.Compute_sen import compute_sen
from copy import deepcopy as copy
# try ray perhaps?
# import ray
# ray.init(num_cpus=4)
# How to use whl files?

import torch
import numpy as np

# A pseudo-random Generator, as recommended by numpy when the code was writen.
rg = Generator(PCG64())
# The gpu that is used in all the rational code (rational part), set to the default gpu that has cuda driver.
cuda = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

space = '---------------------------------------------------------------------------------------------'


# Silence ALL the warning (optional).
# import warnings
# warnings.filterwarnings("ignore")


# Input: data a 2d numpy array, i.e. a table (possible to input a list which would be cast to a numpy array).
# Output: a numpy array that contains the robust mean of every row (mean after removing the highest and lowest 25
# percents of the data); every entry corresponds to the row with the same index.
def robust_mean(data):
    # Change the data to an numpy-array to allow to use numpy functions.
    np_data = np.asarray(np.copy(data))
    # Save in a list the robust mean of every row.
    robust_mean = []
    for cur_data in np_data:
        high = np.percentile(cur_data, 75)
        low = np.percentile(cur_data, 25)
        cur = cur_data[cur_data >= low]
        cur = cur[cur <= high]
        robust_mean.append(cur.mean())
    # Return the new list, cast to a numpy array.
    return np.asarray(robust_mean)


# Input: X, a 1d numpy array, and args, a pair (c,d) of 1d numpy arrays of the same size.
# Output: an array that contains poly(c,x)/( 1+x*poly(d,x) ) for every x in X.
# Identical to return [poly(c,x)/( 1+x*poly(d,x) ) for  x in X], but faster due the use to matrix operations.
# For version that computes this in parallel over various args, i.e. retursns [predict(X, arg) for arg in args],
# but in a faster manner (using matrix operations) see multi_predict.
def predict(X, args):
    c, d = args
    poly_c = np.polyval(np.flip(c), X)
    poly_d = np.polyval(np.flip(d), X)
    return poly_c / (1 + X * poly_d)


# Note that the code bellow uses torch cuda (if there is a supported gpu would use the gpu) for speed up.
# This code bellow is a steam like version of np.polyval, and is inspired from the source code of numpy.
# ''Evaluate a set of polynomial at specific values'' [inspired by the documentation of np.polyval].
# Input: A, a torch cuda 2-D tensor, and X a torch cuda 1-D tensor.
# Output: A torch cuda 2-D tensor that contains [[poly(a,x) for x in X] for a in A], but with a significantly faster
# construction due than this form (at least expected on most machines) to the use of matrix operations,
# and assuming that there is a supported gpu, also gpu speed up.
def multi_polyval(A, X):
    n, k, m = X.__len__(), A.__len__(), A[0].__len__()
    X = X.reshape(1, n).repeat(m, 1)
    A_ = torch.flipud(A)
    A_ = A_.reshape(k, m, 1).repeat(1, 1, n)
    Y = torch.zeros_like(X, dtype=torch.float64, device=cuda)
    for a in A_:
        Y = Y * X + a
    return Y


# Input: X, a 1d numpy array, and args, a 3d numpy array where the second dimension is 2.
# Output: an array that contains poly(c,x)/( 1+x*poly(d,x) ) for every x in X and every (c,d) in args.
# Identical to return [[poly(c,x)/( 1+x*poly(d,x) ) for  x in X] for (c,d) in args] , but faster due the use to matrix
# operations and gpu speed up.
# For a simplified version for len(args)=1, that for this case yields some speed up, see predict.
def multi_predict(X, args):
    # Split the arguments and create 2 cuda tensors from them for the call to multi_polyval.
    n_args = np.array(args)
    C, D = n_args[:, 0, :], n_args[:, 1, :]
    C, D = np.transpose(C), np.transpose(D)
    C, D, X = torch.tensor(C, device=cuda), torch.tensor(D, device=cuda), torch.tensor(X, device=cuda)

    # Calculate the prediction.
    poly_c = multi_polyval(C, X)
    poly_d = multi_polyval(D, X)
    return poly_c / (1 + X * poly_d)


# Input: x,y a (n,2k) numpy arrays.
# Output: A an (n,2k,2k) array, where every A[i] is the matrix defined in Lemma ... (solver construct) for
# P= x[i] \times y[i].
# Done with numpy and not torch since the operations are simple and using the gpu would yield negligable speed up,
# and on the other hand using polyvander from numpy (computes Vandermonde matrix with streaming support, at the time,
# to the best of my knowledge, not supported in PyTorch) yields a significant speed up.
def construct(x, y, k):
    X = np.polynomial.polynomial.polyvander(x, k)  # computes Vandermonde matrix for every x_ in x
    y_ = np.expand_dims(y, 2).repeat(k, 2)
    A_1 = X[:, :, :k]
    A_2 = - y_ * X[:, :, 1:]
    return np.concatenate((A_1, A_2), axis=2)


# Input: k,lim integers, and X,Y (lim,2k) numpy arrays.
# Output: [Solver(x\times y) for (x,y) in zip(X,Y)], where Solver(P) is a pair (c,d)\in R^{k} \times R^{k} such that
# for every (x,y) in P we have poly(c,x)/( 1+x*poly(d,x) ) = y; as defined in the paper.
# As proven in the paper Solver above is well defined and can be computed (as done in the code) for a set of 2k points
# on the plane, but due to machine precision the condition number of the matrices might in practice be extremely large
# (hence, while theoretically the system is solvable, in practice the solution would be "too noisy" to be considered a
# solution and the solver yields an error), and hence, for robustness, we remove the near singular matrices.
def solver(X, Y, k, lim):
    # Create all the systems to solve.
    A = construct(X, Y, k)

    # Remove singular matrices (rarely the occurs, but ignoring this would yield error in those cases).
    # This can be caused by "too much" points in a "too small" range (which causes almost all the values to be the same)
    good = np.linalg.cond(A) < 1e20  # Inspired by https://stackoverflow.com/a/57074564.
    # good = np.linalg.cond(A) < 1e20
    A, Y = A[good], Y[good]

    # Move the matrices to the GPU if there is one (done after removal for speed up).
    A_ = torch.tensor(A, device=cuda, dtype=torch.float64)
    Y_ = torch.tensor(Y, device=cuda, dtype=torch.float64).reshape(-1, 2 * k, 1)
    # Solve the systems (on the GPU if there is one).
    ans = torch.linalg.solve(A_, Y_)

    # Return the result, spited to c and d for simplicity.
    ans = ans.reshape(-1, 2, k).cpu().data.numpy()
    return ans


# Input: X,Y  1d numpy arrays of the same size and a positive integer k.
# Output: All the k-sets of points in X \times Y, split to x and y values (i.e. a pair of 2d numpy arrays).
# Added to support streaming calls to solver above (concatenating the options and calling solver once),
# this is done in Batch_approx and since the computation in solver is using matrix operations (and GPU if there is one)
# this allows for significant increase in the generalization.
def get_centroid_set_samples(X, Y, k):
    subsets = list(combinations(range(len(X)), 2 * k))
    return [[X[i] for i in set] for set in subsets], [[Y[i] for i in set] for set in subsets]


# Input: X,Y  1d numpy arrays of the same size and a positive integer k.
# Output: { solver(Q) \mid Q\subset P, |Q|=k}, where P:=X\times Y.
def centroid_set(X, Y, k):
    X_, Y_ = get_centroid_set_samples(X, Y, k)  # Construct the sets for the solver.
    return solver(np.array(X_), np.array(Y_), k, X_.__len__())  # Return the output of solver.


# Input: X,Y  1d numpy arrays of the same size and 2 positive integers lim,k.
# Output: A sample of lim sets of k points from X \times Y, split to x and y values (i.e. a pair of 2d numpy arrays).
# Added to support streaming calls to solver above; see documentation of get_centroid_set_samples.
def get_Fast_samples_set(X, Y, lim, k):
    n = len(X)
    subsets = [rg.choice(n, 2 * k, False) for _ in range(lim)]
    X_, Y_ = [[X[i] for i in set] for set in subsets], [[Y[i] for i in set] for set in subsets]
    return X_, Y_


# FAST-CENTROID-SET as defined in the paper; see Algorithm 8.
# Input: X,Y  1d numpy arrays of the same size and 2 positive integers lim,k.
# Output: [solver(Q) for Q in G], where G is a sample of lim sets of k points from X \times Y.
def Fast_centroid_set(X, Y, lim, k):
    X_, Y_ = get_Fast_samples_set(X, Y, lim, k)
    return solver(np.array(X_), np.array(Y_), k, lim)


# Input: X,Y a pair of 1d numpy array, and args, a 3d numpy array where the second dimension is 2.
# Output: argmin_{q\in args} \sum_{(x,y))\in X \times Y} |y-predict(x,q) |.
def find_arg_min(X, Y, args):
    pred = multi_predict(X, args).cpu().data.numpy()
    Y_ = np.expand_dims(Y, axis=0).repeat(repeats=pred.__len__(), axis=0)
    miss_mat = np.abs(pred - Y_)
    miss_mat = np.mean(miss_mat, axis=1)
    miss_mat = miss_mat[np.isfinite(miss_mat)]
    if len(miss_mat)==0:
        return np.zeros_like(args[0])
    return args[np.argmin(miss_mat)]


def find_arg_min_weigthed(X, Y, W, args):
    pred = multi_predict(X, args).cpu().data.numpy()
    Y_ = np.expand_dims(Y, axis=0).repeat(repeats=pred.__len__(), axis=0)
    miss_mat = np.abs(pred - Y_) * W
    miss_mat = np.mean(miss_mat, axis=1)
    miss_mat = miss_mat[np.isfinite(miss_mat)]
    if len(miss_mat)==0:
        return np.zeros_like(args[0])
    return args[np.argmin(miss_mat)]


# Practical version of BATCH-APPROX.
def batch_approx(x_split, y_split, k, fast_init, lim=None):
    if lim is None:
        lim = pow(2, k + 5)
    X_, Y_, S = [], [], []
    if fast_init:
        for x, y in zip(x_split, y_split):
            x_, y_ = get_Fast_samples_set(x, y, lim, k)
            X_.extend(x_), Y_.extend(y_), S.append(x_.__len__())
    else:
        for x, y in zip(x_split, y_split):
            x_, y_ = get_centroid_set_samples(x, y, k)
            X_.extend(x_), Y_.extend(y_), S.append(x_.__len__())

    options = solver(np.array(X_), np.array(Y_), k, np.sum(S))

    F, X, Y = [], [], []
    start = 0
    for s, x, y in zip(S, x_split, y_split):
        f = find_arg_min(x, y, options[start:start + s])
        start += s
        F.append(f), X.append(x), Y.append(predict(x, f))
    return F, X, Y


# Heuristic reduce.
def simple_reduce(x,y,k):
    options = Fast_centroid_set(x,y,pow(2,k+5),k)
    return find_arg_min(x, y, options)


# Simplified version of of the main part of Algorithm 7 in the paper, aims to construct an alpha-approximation.
def simplified_streaming_coreset(X, Y, k, beta=pow(2, 10), fast_init=True, reduce_beta=pow(2, 5)):
    x_split, y_split = np.array_split(X, beta), np.array_split(Y, beta)
    F, X, Y = batch_approx(x_split, y_split, k, fast_init)

    while X.__len__() >= 2:
        A = np.array_split(np.arange(0, X.__len__(), 1), np.ceil(X.__len__() / reduce_beta))
        F_, X_ = [], []
        for arr in A:
            x_, y_ = [X[i] for i in arr], [Y[i] for i in arr]
            x, y = np.concatenate(x_, axis=0), np.concatenate(y_, axis=0)
            if arr.__len__() > 1:
                f =simple_reduce(x,y,k)
                F_.append(f), X_.append(x)
            else:
                F_.append(F[np.min(arr)]), X_.append(x)

        Y_ = [predict(x, f) for (x, f) in zip(X_, F_)]
        X.clear(), Y.clear(), F.clear()
        X, Y, F = X_, Y_, F_

    return X[0], Y[0], F[0]


# Poly sample as defined in the paper, added debug to check if there was a significant difference between a uniform
# sample and the sample using the sensitivity (in our test there was not a significant difference).
def poly_sample_robust(X, l, k):
    if len(X) <= l:
        w = np.ones_like(X)
        return X, w
    V = np.polynomial.polynomial.polyvander(X, 2 * k)
    s = compute_sen(V)

    p = s / s.sum()
    n = len(X)
    if np.abs(p.mean() - p).max() > p.mean():
        print('Debug check, there was a point with significantly larger sensitivity.')

    p = s / s.sum()
    samp = rg.choice(n, l, p=p)

    # Compute the frequencies of each sampled item; inspired by ...
    hist = np.histogram(samp, bins=range(n))[0].flatten()
    indxs = np.nonzero(hist)[0]
    w = 1 / (l * p)
    w = w[indxs] * hist[indxs]

    return X[indxs], w


def poly_sample(X, l, k):
    if len(X) <= l:
        w = np.ones_like(X)
        return X, w

    n = len(X)
    p = np.ones(n) / n
    samp = rg.integers(0, n + 1, l)

    # Compute the frequencies of each sampled item; inspired by ...
    hist = np.histogram(samp, bins=range(n))[0].flatten()
    indxs = np.nonzero(hist)[0]
    w = 1 / (l * p)
    w = w[indxs] * hist[indxs]

    return X[indxs], w


def exponential_sample(X, l, k):
    # In this case we will "sample" all the points in X, so simply returning them would improve the performance.
    if len(X) <= 2*l:
        return X, np.ones_like(X)

    m = int(np.floor(np.log2(len(X)))) - 1
    split = np.logspace(1, m, m, base=2).astype(np.int) - 1
    X_split = np.split(X, split)
    X_split2 = np.split(np.flipud(X_split.pop()), split)
    left = X_split2.pop()
    X_split.extend(X_split2)

    if len(left) > 0:
        if len(left) <= len(X) / 4:
            X_split.append(left)
        else:
            mid = int(np.ceil(len(left) / 2))
            X1, X2 = left[:mid], left[-mid:]
            X_split.append(X1), X_split.append(X2)
    S, W = [], []
    for x in X_split:
        s, w = poly_sample(x, l, k)
        S.extend(s), W.extend(w)
    return np.array(S), np.array(W)


def mini_reduce(X, q, l):
    c, c_ = q
    c2 = np.flipud(np.append(np.flipud(c_), 1))
    roots = np.polynomial.polynomial.polyroots(c2)
    ex = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(c2))

    D_ends = np.concatenate((roots, ex))
    D_ends = D_ends[np.isreal(D_ends)]
    if len(D_ends) == 0:
        return exponential_sample(X, l, len(c))

    D_ends = np.concatenate((D_ends, np.array([-np.inf, np.inf])))

    D_ends.sort()
    S, W = [], []
    for (s, e) in zip(D_ends[:-1], D_ends[1:]):
        slice = np.logical_and(s <= X, X < e)
        X_cur = X[slice]
        if len(X_cur) > 0:
            s, w = exponential_sample(X_cur, l, len(c))
            S.extend(s), W.extend(w)

    return np.array(S), np.array(W)


def spesific_reduce(X_arr, F_arr, l, k):
    P, W, P_ = [], [], []
    for X, F in zip(X_arr, F_arr):
        X_i, Y_i = [], []
        for x, f in zip(X, F):
            s, w = mini_reduce(x, f, l)
            y_s = predict(s, f)
            p = list(zip(s, y_s))
            P.extend(p), W.extend(w)

            y = predict(x, f)
            X_i.extend(x), Y_i.extend(y)

        P_.append([np.asarray(X_i), np.asarray(Y_i)])

    P, W = np.array(P), np.array(W)
    options = Fast_centroid_set(P[:, 0], P[:, 1],pow(2,k+5),k)
    q = find_arg_min_weigthed(P[:, 0], P[:, 1],W,options)
    L = [np.abs(predict(x, q) - y).sum() for (x, y) in P_]
    C = np.argpartition(L, -6 * k + 3)[-6 * k + 3:]
    C.sort()
    # print(np.argsort(L).tolist())
    # print(C)
    X_, F_ = [], []
    X_cur = []
    for i, (X, F) in enumerate(zip(X_arr, F_arr)):
        if i in frozenset(C):
            if len(X_cur) > 0:
                X_.append(np.array(X_cur)), F_.append(q)
            for x, f in zip(X, F):
                X_.append(x), F_.append(f)
            X_cur, F_cur = [], []
        else:
            X_cur.extend(np.concatenate(X))
    if len(X_cur) > 0:
        X_.append(np.array(X_cur)), F_.append(q)

    return X_, F_, np.delete(L, C).sum()


# See https://stackoverflow.com/a/11303237 .
def fast_del(L, ind):
    return [v for i, v in enumerate(L) if i not in frozenset(ind)]


def reduce_loss_for_ite(X_arr, F_arr, lam, k, opt):
    X_c, F_c = fast_del(copy(X_arr), opt), fast_del(copy(F_arr), opt)
    return spesific_reduce(copy(X_c), copy(F_c), lam, k)[2]


def expand_options(options, n):
    def expand_opt(opt):
        opt = np.asarray(opt, np.int)
        opt = np.concatenate((opt - 1, opt, opt + 1))
        opt = opt[opt >= 0]
        opt = opt[opt < n]
        opt = np.unique(opt)
        return tuple(np.sort(opt).tolist())  # https://stackoverflow.com/a/51359795 .

    # options2 = Parallel(n_jobs=4)(delayed(expand_opt)(opt) for opt in options)
    options2 = [expand_opt(opt) for opt in options]
    return list(set(options2))  # https://stackoverflow.com/a/27305828 .


def reduce(X_arr, F_arr, lam, k):
    # print(comb(len(X_arr),2 * k - 1,True))
    # options_ = list(combinations(range(len(X_arr)), 2 * k - 1))
    # options_ = [[i] for i in range(len(X_arr))]
    options_ = [rg.choice(len(X_arr), 2 * k-1, False) for _ in range(32)]
    options = expand_options(options_, len(X_arr))
    # losses = [reduce_loss_for_ite(X_arr, F_arr, lam, k,opt) for opt in options]
    losses = Parallel(n_jobs=4)(delayed(reduce_loss_for_ite)(copy(X_arr), copy(F_arr), lam, k, opt) for opt in frozenset(options))
    ind = np.argmin(losses)
    opt = options[ind]

    X_c, F_c = fast_del(X_arr, opt), fast_del(F_arr, opt)
    X_, F_, c = spesific_reduce(X_c, F_c, lam, k)

    X, F = X_, F_
    for i, (x1, f1) in enumerate(zip(X_arr, F_arr)):
        if i in frozenset(opt):
            for x2, f2 in zip(x1, f1):
                X.append(x2), F.append(f2)

    return X, F


# The main part of Algorithm 7 in the paper, aims to construct an (alpha,beta)-approximation.
def streaming_coreset(X, Y, k, beta=pow(2, 10), fast_init=True, lim=None, reduce_beta=pow(2, 5),spesific=False):
    if lim is None:
        lim = pow(2,k+5)
    x_split, y_split = np.array_split(X, beta), np.array_split(Y, beta)
    F, X, _ = batch_approx(x_split, y_split, k, fast_init)
    X, F = [[x] for x in X], [[f] for f in F]
    while X.__len__() >= 2:
        A = np.array_split(np.arange(0, X.__len__(), 1), np.ceil(X.__len__() / reduce_beta))
        F_, X_ = [], []
        for arr in A:
            # for i, arr in enumerate(A):
            # print('Starting on', i+1, 'out of', len(A), '.')
            x, f = [X[i] for i in arr], [F[i] for i in arr]
            if arr.__len__() > 1:
                if spesific:
                    x, f, _ = spesific_reduce(x, f, lim, k)
                else:
                    x, f = reduce(x, f, lim, k)
                F_.append(f), X_.append(x)
            else:
                F_.append(F[np.min(arr)]), X_.append(x)

        X.clear(), F.clear()
        X, F = X_, F_

    return X[0], F[0]



def Bi_pred(X,F):
    args = [[(x, f), x[0]] for (x, f) in zip(X, F)]
    args = sorted(args, key=lambda x: x[1])
    Y = []
    for [(x, f),_] in args:
        Y.extend(predict(x, f))

    return np.asarray(Y)


def coreset_sample(X, Y, CoreSetSize, miss, pred):
    p = miss / miss.sum()

    n = len(X)
    samp = rg.choice(n, CoreSetSize, True, p=p)
    # Compute the frequencies of each sampled item; inspired by ...
    hist = np.histogram(samp, bins=range(n))[0].flatten()
    indxs = np.nonzero(hist)[0]
    W = 1 / (p * CoreSetSize)
    W = W[indxs] * hist[indxs]

    C = Y[indxs]
    D = pred[indxs]

    return indxs, C, D, W


def improve_approx_true_heuristic(X, Y, f, eps):
    pred, [c, d] = predict(X, f), f

    beta = np.ceil(np.log(1.1) / np.log(1 + eps)).__int__()
    s = (2 * beta + 1) ** (2 * len(c))
    if s > pow(10, 4):
        # print(s, eps)
        return improve_approx_true_heuristic(X, Y, f, eps * 1.1)

    C = np.logspace(1, beta, beta, base=1 + eps) - 1

    C_ = np.ones((2 * beta + 1,))
    C_[:beta] = 1 - np.flip(C)
    C_[beta + 1:] = 1 + C

    C = product(C_, repeat=2 * c.__len__())
    C = np.array(list(C))
    C = np.stack(np.array_split(C, 2, 1), 1)

    F = np.expand_dims(np.array(f), 0) * C

    F_ = np.array_split(F, 10)
    F = [find_arg_min(X, Y, f_) for f_ in F_]
    return find_arg_min(X, Y, F)


from time import perf_counter as time

if __name__ == '__main__':
    t = time()

    n, k = pow(2, 15), 1
    for _ in range(100):
        c_t = time()
        X = np.linspace(0, 2, n)
        q = np.ones((2, k))  # rg.uniform(0, 1, 2 * k).reshape((2, k))
        Y = predict(X, q) + rg.uniform(-0.1, 0.1, n)
        miss = streaming_coreset(X, Y, k)
        print('Algorithm loss:', miss)
        print('GT loss:', np.abs(predict(X, q) - Y).sum() / n)
        print('Time:', np.round(time() - c_t, 5))
        c_t = time()
        func = simplified_streaming_coreset(X, Y, k)[2]
        miss2 = np.abs(predict(X, func) - Y).sum() / n
        print('Algorithm 2 loss:', miss2)
        print('Algorithm 2 Time:', np.round(time() - c_t, 5))
        print(space)

    print(np.round(time() - t, 3))
