import csv
import numpy as np
from time import perf_counter as time
from Coreset_airQ import test, print_res_full,example_vs_scipy,plot_res_time
import matplotlib.pyplot as plt
import json
import glob
import errno, os, stat, shutil
from copy import deepcopy as copy

# Extraction parameters
path = '../Datasets/Beijing/Beijing_csv'
datasets_folder = '../Datasets/Beijing/save_folder'

# Input and output parameters.
station = 'Aotizhongxin'

save_folder = '../Results/Beijing/Aotizhongxin/Worst'
input_based = True
input_folder = '../Results/Beijing/Aotizhongxin/OPT'
# 'opt' for approx to optimal query, default is worst case over the queries.
global_test_type = 'opt'


# https://stackoverflow.com/a/1214935, some times it is extremely hard to delete files.
def handleRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)
    else:
        raise


def extract():
    for f in glob.iglob(save_folder + '\\*'):
        shutil.rmtree(f, ignore_errors=False, onerror=handleRemoveReadonly)

    for filepath in glob.iglob(path + '\\*.csv'):
        arr = None
        names = None
        t = time()
        with open(filepath, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row[1].__eq__('2017'):
                    break
                if arr is None:
                    arr = [[] for _ in range(4)]
                    names = copy(row)
                else:
                    j = 0
                    for i, v in enumerate(row):
                        if i == 1 or (i > 10 and i < 14):
                            print(names[i])

                            if v.__eq__('NA'):
                                arr[j].append(np.inf)
                            else:
                                arr[j].append(np.float(v))
                            j += 1

        Y = np.array(arr)
        years = np.unique(Y[0, :])
        name_p_csv = str.split(filepath, '\\').pop()
        name = str.split(name_p_csv, '.').pop(0)
        cur = datasets_folder + '\\' + name
        os.mkdir(cur)
        for i, year in enumerate(years):
            Y_cur = Y[:, Y[0, :] == year]
            for j in range(1, 4):
                y = Y_cur[j]
                Y_cur[j, np.isinf(y)] = np.mean(y[np.isfinite(y)])

            np.save(cur + '\\year ' + str(int(year)) + '.npy', Y_cur[1:, :], allow_pickle=True)
        print(time() - t)


#   def example():
#       for filepath in glob.iglob(datasets_folder + '\\*.npy'):
#           Y = np.load(filepath, allow_pickle=True)
#           name = str.split(filepath, '\\').pop()
#           for i, y in enumerate(Y):
#               X = np.linspace(1, len(y), len(y))
#               plt.scatter(X, y, c='r')
#               plt.title(name + str(i))
#               plt.show()


def run_test_one_dataset(titles, Y, X, CS, k, test_num, runs):
    st = time()
    test(X, Y, k, global_test_type, runs=runs)
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
    print(time() - st)
    return np.array(M), np.array(T), np.array(S), np.array(B)


def run_test(titles, CS, k, test_num=100, runs=1):
    for f in glob.iglob(save_folder + '\\*\\**'):
        os.remove(f)
    for f in glob.iglob(save_folder + '\\*'):
        shutil.rmtree(f, ignore_errors=False, onerror=handleRemoveReadonly)

    for folder in glob.iglob(datasets_folder + '\\*'):
        name = str.split(folder, '\\').pop()
        if name.__eq__(station):
            for file in glob.iglob(folder + '\\*.npy'):
                year_npy = str.split(file, '\\').pop()
                year = str.split(year_npy, '.').pop(0)
                cur_save_folder = save_folder + '\\' + year
                os.mkdir(cur_save_folder)
                year = int(year[4:])
                print(year)
                Y = np.load(file)
                for i, y in enumerate(Y):
                    print('Starting on entry', i + 1, 'out of', str(len(Y)) + ', for year', year,
                          'out of 2016, for folder', name)

                    x = np.linspace(1, len(y), len(y))

                    M, T, S, B = run_test_one_dataset(titles, y, x, CS, k, test_num, runs)
                    path = cur_save_folder + '\\entry ' + str(i + 1) + '.txt'

                    file = open(path, mode='w', newline='\n', encoding="utf8")
                    json.dump([M.tolist(), T.tolist(), S.tolist(), B.tolist()], file)
                    file.close()


def plot_res(M, S, titles, title):
    # if not 'PRES' in title:
    #    return
    colors = ['r', 'b', 'g', 'k', 'm', 'c']
    M, S = np.array(M), np.asarray(S)
    max_, min_ = 0, np.inf
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
    mng.window.size()
    mng.window.showMaximized()
    max_ = min(max_,8) if global_test_type.__eq__('opt') else min(max_,15)
    if 'PRES' in title:
        max_ = min(max_,5) if global_test_type.__eq__('opt') else min(max_,12)
    plt.ylim(min_, max_)

    plt.title(title)
    plt.legend()
    plt.show()



def plot_results(titles, entry_dict):
    for year_folder in glob.iglob(input_folder + '\*'):
        year = str.split(year_folder, '\\').pop()
        year_num = int(str.split(year, ' ').pop())
        for entry_path in glob.iglob(year_folder + '\*'):
            entry_npy = str.split(entry_path, '\\').pop()
            entry = str.split(entry_npy, '.').pop(0)
            entry_num = int(entry[-1])
            title = 'The plot for ' + entry_dict[entry_num] + ' for ' + year + '.'
            file = open(entry_path, newline='\n', encoding="utf8")
            M, T, S, B = json.load(file)
            file.close()

            # if year_num == 2016:
            plot_res(M, S, titles, title)
            # if "TEMP for year 2016" in title:
            #    plot_res_time(T, S, titles,title)

import latex
def example(entry_dict):
    for folder in glob.iglob(datasets_folder + '\\*'):
        name = str.split(folder, '\\').pop()
        if name.__eq__(station):
            for file in glob.iglob(folder + '\\*.npy'):
                year_npy = str.split(file, '\\').pop()
                year = str.split(year_npy, '.').pop(0)
                year = int(year[4:])

                Y = np.load(file)
                print(year, 'point count',len(Y[0]))
                for i, y in enumerate(Y):
                    print('Starting on entry', i + 1, 'out of', str(len(Y)) + ', for year', year,'out of 2016, for folder', name)
                    print('')

                    x = np.linspace(1, len(y), len(y))
                    title = entry_dict[i + 1]+' for year ' + str(year)
                    example_vs_scipy(x, y,title)


if __name__ == '__main__':
    # extract()
    titles = ['Our-RFF', 'Our-FRFF', 'Gradient', r'$\ell_{\infty}$-Coreset', 'RandomSample', 'NearConvexCoreset']
    entry_dict = {1: "TEMP", 2: 'PRES', 3: 'DEWP'}
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'lines.linewidth':5})
    # example(entry_dict)

    # 7,7.25,7.5,7.75,8,8.25,8.5,8.75,9,9.25,9.5,9.75,10
    CS = np.logspace(7, 10, 13, base=2).round().astype(np.int)
    k = 2
    if input_based:
        plot_results(titles, entry_dict)
    else:
        run_test(titles, CS, k)
