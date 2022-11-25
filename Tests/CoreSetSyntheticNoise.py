import numpy as np
from time import perf_counter as time
from algorithms import rg
from Helpers.loss import test_core
import json
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from algorithms import simplified_streaming_coreset, predict
from GenOpts import minimize, sci_loss

# Input and output parameters.
save_path = '../Results/Synthetic/opt/save.txt'
input_based = True
input_path = '../Results/Synthetic/def/save.txt'
space = '---------------------------------------------------------------------------------------------'
# opt for for approx to optimal query, default is worst case over the queries.
global_test_type = 'def'


def generate_options_fib(s, p, count, prec=99, add_noise=True):
    X = np.linspace(-0.5, 0.5, count)
    seq = list(s)
    for i in range(prec):
        cur = np.array(seq)
        last = cur[i:]
        seq.append(np.sum(p * last))

    seq = np.array(seq)
    if add_noise:
        seq += rg.normal(0, 1 / 4, len(seq))
    Y = np.polyval(np.flip(seq), X)
    return X, Y


def get_par(s, p):
    k = len(p)
    d = -np.flip(p)
    C = Polynomial(s)
    for i in range(k):
        cur = s[:i] * p[i]
        padding = np.zeros(k - len(cur))
        cur = np.concatenate([padding, cur])

        C = C - Polynomial(cur)
        c = C.coef
        if len(c) < k:
            padding = np.zeros(k - len(c))
            C = Polynomial(np.concatenate([c, padding]))

    return C.coef, d


def test(n, test_type, runs, k, smp_size):
    s, p = np.ones(k), np.ones(k)
    X, Y = generate_options_fib(s, p, n)
    X = np.linspace(1, len(Y), len(Y))
    return test_core(X, Y, k, runs, smp_size, test_opt=test_type.__eq__('opt'))


def get_stats_sting(titles, X, s=''):
    for (t, x) in zip(titles, X):
        if len(s) > 0:
            s += ' '
        s = s + t + ' = ' + str(x) + ';'
    return s[:-1] + '.'


def print_res(M, S, DS, titles):
    for j, n in enumerate(DS):
        m = np.mean(M[j], 0).round(2)
        print('n =', str(n) + ', mean loss:', get_stats_sting(titles, m))
    print(space)


def print_res_partial(X, titles, t):
    print(t, 'mean:', get_stats_sting(titles, X.mean(0).round(2)))
    print(t, '25 %:', get_stats_sting(titles, np.percentile(X, 25, 0).round(2)))
    print(t, '75 %:', get_stats_sting(titles, np.percentile(X, 75, 0).round(2)))
    print(space)


def print_res_full(M, T, S, B, DS, titles):
    for m, t, s, b, n in zip(M, T, S, B, DS):
        print('Results for n = ', n)
        print_res_partial(np.asarray(m), titles, 'Loss')
        print(space)


def int_to_roman(number):  # https://stackoverflow.com/a/47713392
    ROMAN = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"), ]

    result = ""
    for (arabic, roman) in ROMAN:
        (factor, number) = divmod(number, arabic)
        result += roman * factor
    return '(' + result + ')'


def plot_res(M, S, B, DS, titles):
    M, S, B = [np.asarray(V).T for V in [M, S, B]]
    colors = ['r', 'b', 'g', 'k', 'm', 'c', 'gray']
    DS = np.asarray(DS)
    max_ = 0
    for i, name in enumerate(titles):
        cur = M[i].T

        s1, s2 = np.percentile(cur, 25, 1).reshape((1, -1)), np.percentile(cur, 75, 1).reshape((1, -1))
        if global_test_type.__eq__('opt') or i < 4:
            max_ = max(s2.max(), max_)
        m, s = np.median(cur, 1), np.concatenate([s1, s2], 0)
        if i == 3:
            cur = [c[c < 100] for c in cur]
            print([len(c) for c in cur])
            m = [np.median(c) for c in cur]
        s = np.abs(s - m)

        plt.errorbar(np.log2(DS), m, s, c=colors[i], label=name, ls='--', marker='s', mfc=colors[i])

    plt.xlabel(r'$\log_2(n)$-where $n$'+ ' is the input signal\'s length')
    plt.ylabel('Approximation error in percents')
    size = S.mean().round().astype(np.int)
    plt.title('Coreset\'s size = ' + str(size))
    plt.legend()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.ylim(0, max_)

    plt.show()


def plot_res_time(T, S, B, DS, titles):
    T, S, B = [np.asarray(V).T for V in [T, S, B]]
    colors = ['r', 'b', 'g', 'k', 'm', 'c']
    for i, name in enumerate(titles):
        cur = T[i]

        s1, s2 = np.percentile(cur, 25, 0).reshape((1, -1)), np.percentile(cur, 75, 0).reshape((1, -1))
        t, s = np.median(cur,0), np.concatenate([s1, s2], 0)
        s = np.abs(s - t)

        plt.errorbar(np.log2(DS), t, s, c=colors[i], label=name, ls='--', marker='s', mfc=colors[i])

    plt.xlabel(r'$\log_2(n)$-where $n$'+ ' is the input signal\'s length')
    plt.ylabel('Construction time (seconds)')
    size = S.mean().round().astype(np.int)
    plt.title('Coreset\'s size = ' + str(size))
    plt.legend()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.ylim(0, 2)
    plt.show()


def run_test(titles, DS, k, smp_size, test_num=100, runs=10):
    M, T, S, B = [[[] for n in DS] for _ in range(4)]

    batch_size = 1
    c_t = time()
    test(DS[0], global_test_type, runs, k, smp_size)
    print('Setup done, took', str(np.round(time() - c_t, 2)), 'seconds.')
    print('Starting actual test:')
    print(space)

    c_t = time()
    for i in range(test_num):
        for j, n in enumerate(DS):
            m, t, s, b = test(n, global_test_type, runs, k, smp_size)
            M[j].append(m), T[j].append(t), S[j].append(s), B[j].append(b)

        if (i + 1) % batch_size == 0:
            print('Done on', np.round((i + 1) / test_num * 100, 2), '% last steps took', np.round(time() - c_t, 2))
            print_res(M, S, DS, titles)
            c_t = time()

    return [np.asarray(v).tolist() for v in [M, T, S, B]]


# sci_loss_wighted
def sci_loss_wighted(q, *args):
    X, Y, W, k = args
    c, d = q[:k], q[k:]
    return (W * np.abs(predict(X, [c, d]) - Y)).sum()


def example_vs_scipy(X, Y, title=None, k=2):
    plt.scatter(X, Y, c='r',label='GT')

    f = minimize(sci_loss, np.zeros(2 * k), (X, Y, k)).x
    sci_fit = f.reshape((2, k))
    miss = np.abs(predict(X, sci_fit) - Y).mean().round(4)
    print(miss)
    plt.plot(X, predict(X, sci_fit), c='b', linestyle='-', label='Scipy')

    beta = pow(2, k + 3)
    q = simplified_streaming_coreset(X, Y + rg.normal(0, 1e-5, len(Y)), k, beta=beta, reduce_beta=beta)[2]
    miss = np.abs(predict(X, q) - Y).mean().round(4)
    print(miss)
    plt.plot(X, predict(X, q), c='k', linestyle='-', label='Ours')

    poly_fit = np.polyfit(X, Y, 2 * k - 1)
    miss = np.abs(np.polyval(poly_fit,X) - Y).mean().round(4)
    print(miss)
    plt.plot(X, np.polyval(poly_fit,X), c='g', linestyle='-', label='Poly-fit')

    #    ind, C, D, w = coreset_sample(X, Y, pow(2, 10), np.abs(predict(X, q) - Y), predict(X, q))
    #    W = np.ones_like(X)
    #    W_ = np.concatenate([W, w, -w])
    #    X_ = np.concatenate([X, X[ind], X[ind]])
    #    Y_ = np.concatenate([Y, C, D])
    #
    #    f = minimize(sci_loss_wighted, np.zeros(2 * k), (X_, Y_, W_, k)).x
    #    sci_fit = f.reshape((2, k))
    #    miss = np.abs(predict(X, sci_fit) - Y).mean().round(4)
    #    plt.plot(X, predict(X, sci_fit), c=(0.5, 0.9, 0.9), linewidth=3, linestyle='--',
    #             label='SciPy-core-pred, avg diff = ' + str(miss))
    #
    #    opt = Fast_centroid_set(X_, Y_, pow(2, 10), k)
    #    q = find_arg_min_weigthed(X_, Y_, W_, opt)
    #    miss = np.abs(predict(X, q) - Y).mean().round(4)
    #    plt.plot(X, predict(X, q), c=(0, 1, 0), linewidth=3, linestyle='--', label='Our-core-pred, avg diff = ' + str(miss))
    #
    plt.xlabel('Index')
    plt.ylabel('Value')
    if title is not None:
        plt.title(title)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    #    plt.title(title)
    plt.legend()
    plt.show()


def example(k):
    s, p = np.ones(k), np.ones(k)
    X, Y = generate_options_fib(s, p, 3 * pow(2, 12))
    X = np.linspace(1, len(Y), len(Y))
    example_vs_scipy(X, Y)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'lines.linewidth':5})
    titles = ['Our-RFF', 'Our-FRFF', 'Gradient', r'$\ell_{\infty}$-Coreset', 'RandomSample', 'NearConvexCoreset']

    # Load data/run tests (according to the value of input_based).
    DS = np.logspace(12, 16, 5, base=2).round().astype(np.int)
    print(DS)
    k = 2

    sizes = [128, 256, 512, 1024]
    # example(k)

    if input_based:
        file = open(input_path, newline='\n', encoding="utf8")
        M, T, S, B = json.load(file)
        print('Loaded data')
        for i, sample_size in enumerate(sizes):
            plot_res_time(T[i], S[i], B[i], DS, titles)
            # plot_res(M[i], S[i], B[i], DS, titles)
    else:
        M, T, S, B = [], [], [], []
        for sample_size in sizes:
            print('Starting on', sample_size)
            Mi, Ti, Si, Bi = run_test(titles, DS, k, sample_size)
            M.append(Mi), T.append(Ti), S.append(Si), B.append(Bi)

        for _ in range(15):
            print(''.join(['Test Done !!! ' for _ in range(16)]))
        print(space), print(space)

        file = open(save_path, mode='w', newline='\n', encoding="utf8")
        json.dump([M, T, S, B], file)
        file.close()
