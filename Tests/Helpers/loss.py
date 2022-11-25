from time import perf_counter as time
from algorithms import Fast_centroid_set, multi_predict, predict,find_arg_min
import numpy as np
from GenOpts import compute_options, core_sample1, core_sample2


def loss_calc1(Checks_pred, GT, P_pred, sample, C, D, W):
    pred_on_sample = Checks_pred[:, sample]

    coreset_loss = np.abs(Checks_pred - P_pred).sum(axis=1)
    c_diff = np.abs(pred_on_sample - C)
    d_diff = np.abs(pred_on_sample - D)
    coreset_loss += np.sum(W * c_diff, axis=1)
    coreset_loss -= np.sum(W * d_diff, axis=1)

    miss = np.abs(coreset_loss - GT) / GT

    return 100 * np.max(miss)


def loss_calc2(Checks, GT, x, y, w):
    coreset_loss = [(w * np.abs(predict(x, f) - y)).sum() for f in Checks]
    miss = 100 * np.abs(np.asarray(coreset_loss) - GT) / GT

    return np.max(miss)


def get_res_worst(X, Y, k, GT, Checks_pred, Checks, approx_arr, sen, smp_size, runs):
    T, M, S = np.zeros((6, runs)), np.zeros((6, runs)), np.zeros((6, runs))
    for i in range(runs):
        t = time()
        [P_pred, sample, C, D, W], s1 = core_sample1(X, Y, approx_arr[0][1], smp_size - 6 * k ** 2, approx_arr[0][0])
        T[0, i] = time() - t
        M[0, i] = loss_calc1(Checks_pred, GT, P_pred, sample, C, D, W)

        t = time()
        [P_pred, sample, C, D, W], s2 = core_sample1(X, Y, approx_arr[1], smp_size)
        T[1, i] = time() - t
        M[1, i] = loss_calc1(Checks_pred, GT, P_pred, sample, C, D, W)

        t = time()
        [P_pred, sample, C, D, W], s3 = core_sample1(X, Y, approx_arr[2], smp_size)
        T[2, i] = time() - t
        M[2, i] = loss_calc1(Checks_pred, GT, P_pred, sample, C, D, W)

        t = time()
        [P_pred, sample, C, D, W], s4 = core_sample1(X, Y, approx_arr[3], smp_size)
        T[3, i] = time() - t
        M[3, i] = loss_calc1(Checks_pred, GT, P_pred, sample, C, D, W)

        t = time()
        I = np.ones_like(X)
        [x, y, w], s5 = core_sample2(X, Y, I, np.round(smp_size * 1.5).astype(np.int))
        T[4, i] = time() - t
        M[4, i] = loss_calc2(Checks, GT, x, y, w)

        t = time()
        [x, y, w], s6 = core_sample2(X, Y, sen, np.round(smp_size * 1.5).astype(np.int))
        T[5, i] = time() - t
        M[5, i] = loss_calc2(Checks, GT, x, y, w)

        S[:, i] = np.array([s1, s2, s3, s4, s5, s6])

    return T, M, S


def gen_query_set(X, Y, k,test_opt):
    Checks = Fast_centroid_set(X, Y, 1024, k)

    if test_opt:
        Checks = find_arg_min(X,Y,Checks).reshape((1,2, k))

    Checks_pred = multi_predict(X, Checks).cpu().data.numpy()
    GT = np.abs(Checks_pred - Y).sum(axis=1)
    ind = np.isfinite(GT)
    return GT[ind], Checks_pred[ind], Checks[ind]


def test_core(X, Y, k, runs, smp_size,test_opt=False):
    approx_arr, sen, T, b = compute_options(X, Y, k, runs, 1)

    GT, Checks_pred, Checks = gen_query_set(X, Y, k,test_opt)

    T_add, M, S = get_res_worst(X, Y, k, GT, Checks_pred, Checks, approx_arr, sen, smp_size, runs)
    T = T[:, 0, :] + T_add

    return M.mean(1), T.mean(1), S.mean(1), b


