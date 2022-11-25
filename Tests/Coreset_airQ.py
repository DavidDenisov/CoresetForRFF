import csv
import numpy as np
from time import perf_counter as time
import json
import matplotlib.pyplot as plt
from CoreSetSyntheticNoise import space, get_stats_sting,example_vs_scipy
from GenOpts import compute_options
from Helpers.loss import gen_query_set,get_res_worst
# Extraction parameters
path = '../Datasets/airQ/AirQualityUCI.csv'
datasets_folder = '../Datasets/airQ/save_folder'
save_folder = '../Results/airQ'

# Input and output parameters.
save_path = '../Results/Actual_Results/airQ/OPT'
input_based = True
input_path = '../Results/airQ/OPT'
# 'opt' for approx to optimal query, default is worst case over the queries.
global_test_type = 'opt'


def extract():
    arr = None
    t = time()
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            if row[3].__eq__(''):
                break
            if arr is None:
                arr = [[] for _ in range(len(row) - 4)]
            else:
                for i, v in enumerate(row):
                    if i not in frozenset([0, 1, 15, 16]):
                        v = v.replace(',', '.')
                        arr[i - 2].append(np.float(v))
    print(time() - t)

    j = 0
    for i, v in enumerate(arr):
        A = np.array(v)
        np.save(datasets_folder + '\\' + str(j) + '.npy', A)
        j += 1


def test(X, Y, k,test_type, CS=None, runs=10):
    if CS is None:
        CS = [256]

    approx_arr, sen, T, b = compute_options(X, Y, k, runs,len(CS))
    GT, Checks_pred, Checks = gen_query_set(X, Y, k, test_opt=test_type.__eq__('opt'))

    M, S = np.zeros_like(T), np.zeros_like(T)
    for i, smp_size in enumerate(CS):
        T_add, M_i, S_i = get_res_worst(X, Y, k, GT, Checks_pred, Checks, approx_arr, sen, smp_size, runs)
        T[:, i, :] += T_add
        M[:, i, :] = M_i
        S[:, i, :] = S_i
    return M.mean(2), T.mean(2), S.mean(2)/len(Y), b


def print_res_partial(X, titles, t):
    print(t, 'mean:', get_stats_sting(titles, X.mean(0).round(2)))
    print(t, 'median:', get_stats_sting(titles, np.median(X, 0).round(2)))
    # print(t, 's.t.d.:', get_stats_sting(titles, np.std(X, (0, 3)).round(2)))
    # print(space)


def print_res_full(M, T, S, B, titles):
    print(space)
    print_res_partial(np.array(M), titles, 'Loss')
    # print_res_partial(np.array(T), titles, 'Time')
    # print_res_partial(np.array(S), titles, 'Sizes')

    # print('Beta Mean = ' + str(np.mean(B).round(4)))
    # print('Beta s.t.d. = ' + str(np.std(B).round(4)))
    print(space)


def run_test(titles, Y, X, CS, k, test_num=100, runs=1):
    test(X, Y, k, global_test_type)
    M, T, S, B = [[] for _ in range(4)]
    c_t = time()
    for i in range(test_num):
        m, t, s, b = test(X, Y, k, global_test_type, CS, runs)
        M.append(m), T.append(t), B.append(b), S.append(s)

        if (i + 1) % 1 == 0:
            print('Done on check number', i + 1, 'out of', test_num)
            print_res_full(M, T, S, B, titles)
            print('Last batch took', np.round(time() - c_t, 2))
            c_t = time()

    return np.array(M), np.array(T), np.array(S), np.array(B)


def plot_res(M, S, titles, title=None):
    colors = ['r', 'b', 'g', 'k', 'm', 'c']
    M,S = np.array(M),np.asarray(S)

    max_,min_ = 0,np.inf
    for i, name in enumerate(titles):
        cur= M[:, i, :]

        s1, s2 = np.percentile(cur, 25, 0).reshape((1, -1)), np.percentile(cur, 75, 0).reshape((1, -1))
        min_ = min(s1.min(), min_)
        if global_test_type.__eq__('opt') or i < 4:
            max_ = max(s2.max(), max_)
        m, s = np.median(cur,0), np.concatenate([s1, s2], 0)
        s = np.abs(s - m)
        s_ = np.median(S[:, i, :],0)
        plt.errorbar(s_, m, s, c=colors[i], label=name, ls='--', marker='s', mfc=colors[i])

    plt.ylabel('Approximation error in percents')
    plt.xlabel('Coreset\'s size in percents of original data')

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    max_ = min(max_, 7) if global_test_type.__eq__('opt') else min(max_, 14)
    plt.ylim(min_, max_)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()


def plot_res_time(T, S, titles,title=None):
    T, S = [np.asarray(V) for V in [T, S]]
    colors = ['r', 'b', 'g', 'k', 'm', 'c']
    for i, name in enumerate(titles):
        cur = T[:,i,:]

        s1, s2 = np.percentile(cur, 25, 0).reshape((1, -1)), np.percentile(cur, 75, 0).reshape((1, -1))
        t, s = cur.mean(0), np.concatenate([s1, s2], 0)
        # s = np.abs(s - t)
        # s_ = 100 * S[:, i, :].mean(0)
        # plt.errorbar(s_, t, s, c=colors[i], label=name, ls='--', marker='s', mfc=colors[i])
        print(name+ ': mean: '+str(t.mean().round(4))+', s.t.d.: '+str(t.std().round(4)))

    #   plt.xlabel('Coreset size in percents of original data')
    #   plt.ylabel('Construction time (seconds)')
    #   plt.legend()
    #   mng = plt.get_current_fig_manager()
    #   mng.window.showMaximized()
    #   plt.ylim(0.0, 1.2)
    #   if title is not None:
    #       plt.title(title)
    #   plt.show()

def example(num):
    Y = np.load(datasets_folder + '\\' + str(num) + '.npy')
    Y[Y < -100] = np.mean(Y[Y > -100])
    X = np.linspace(1,len(Y), len(Y))
    if num ==12:
        S = 'Air quality example, Humidity'
    else:
        S = 'Air quality example, Temp'
    example_vs_scipy(X, Y, S, k=2)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'lines.linewidth':5})
    # example(10)
    # example(12)

    titles = ['Our-RFF', 'Our-FRFF', 'Gradient', r'$\ell_{\infty}$-Coreset', 'RandomSample', 'NearConvexCoreset']
    # Log scale, will not be equal to a linear interpolation between the points.
    CS = np.logspace(7, 10, 13, base=2).round().astype(np.int)
    # CS = np.logspace(7, 10, 4, base=2).round().astype(np.int)
    print(CS)
    # Load data/run tests (according to the value of input_based).
    if input_based:
        for entry in [10, 12]:
            file = open(input_path+ '\\' + str(entry) + '.txt', newline='\n', encoding="utf8")
            M, T, S, B = json.load(file)
            file.close()
            Y = np.load(datasets_folder + '\\' + str(entry) + '.npy')

            # plot_res(M, S, titles)
            if entry ==12:
                plot_res_time(T, S, titles)
    else:
        for entry in [10, 12]:
            Y = np.load(datasets_folder + '\\' + str(entry) + '.npy')
            Y[Y < -100] = np.mean(Y[Y > -100])
            X = np.linspace(1,len(Y), len(Y))
            k = 2

            print(len(Y))
            print(CS[0] - 6 * k ** 2)

            M, T, S, B = run_test(titles, Y, X, CS, k)

            file = open(save_path + '\\' + str(entry) + '.txt', mode='w', newline='\n', encoding="utf8")
            json.dump([M.tolist(), T.tolist(), S.tolist(), B.tolist()], file)
            file.close()

            # plot_res(M, S, titles)

        for _ in range(15):
            print(''.join(['Test Done !!! ' for _ in range(16)]))
