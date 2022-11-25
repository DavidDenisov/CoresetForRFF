from algorithms import predict, coreset_sample, batch_approx, rg, Fast_centroid_set, find_arg_min_weigthed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# sci_loss_wighted
def sci_loss_weighed(q, *args):
    X, Y, W, k = args
    c, d = q[:k], q[k:]
    return (W * np.abs(predict(X, [c, d]) - Y)).sum()


# Heuristic reduce.
def simple_reduce(x, y, w, k):
    options = Fast_centroid_set(x, y, pow(2, k + 5), k)
    return find_arg_min_weigthed(x, y, w, options)


# Simplified version of of the main part of Algorithm 7 in the paper, aims to construct an alpha-approximation.
def simplified_streaming_coreset_weighed(X, Y, W, k):
    beta = pow(2, k + 3)
    x_split, y_split, W = np.array_split(X, beta), np.array_split(Y, beta), np.array_split(W, beta)
    F, X, Y = batch_approx(x_split, y_split, k, True)

    while X.__len__() >= 2:
        A = np.array_split(np.arange(0, X.__len__(), 1), np.ceil(X.__len__() / beta))
        F_, X_, W_ = [], [], []
        for arr in A:
            x_, y_, w_ = [X[i] for i in arr], [Y[i] for i in arr], [W[i] for i in arr]
            x, y, w = np.concatenate(x_, axis=0), np.concatenate(y_, axis=0), np.concatenate(w_, axis=0)
            X_.append(x), W_.append(w)
            if arr.__len__() > 1:
                f = simple_reduce(x, y, w, k)
                F_.append(f)
            else:
                F_.append(F[np.min(arr)])

        Y_ = [predict(x, f) for (x, f) in zip(X_, F_)]
        X.clear(), Y.clear(), F.clear(), W.clear()
        X, Y, F, W = X_, Y_, F_, W_

    return F[0]


def gen_opt(X,Y,W,k):
    poly_fit = np.polyfit(X, Y, 2 * k - 1)

    # np.zeros(2 * k) rg.normal(0,1e-5,2 * k)
    f = minimize(sci_loss_weighed, np.zeros(2 * k), (X, Y, W, k)).x
    sci_fit = f.reshape((2, k))

    ours = simplified_streaming_coreset_weighed(X, Y + rg.normal(0, 1e-5, len(Y)), W, k)

    return poly_fit,sci_fit,ours


def plot_with_bars(axes,X,F,c,linestyle,label,poly=False):
    if poly:
        Y = [np.polyval(f, X) for f in F]
    else:
        Y = [predict(X,f) for f in F]
    axes.plot(X, np.mean(Y,0), c=c, linestyle=linestyle, label=label)
    axes.fill_between(X,np.min(Y,0),np.max(Y,0), color=c,alpha=0.5)

def plot(X,Y,poly_fit, sci_fit, ours ):
    fig, axes = plt.subplots()
    axes.scatter(X, Y, c='r', label='GT')
    plot_with_bars(axes,X,poly_fit,poly=True, c=(0,1,0), linestyle='--', label='Poly-fit')
    plot_with_bars(axes, X, sci_fit, c=(0,0,1), linestyle='-.', label='Gradient')
    plot_with_bars(axes, X, ours, c='k', linestyle='--', label='Ours')
    # axes.plot(X, np.polyval(poly_fit,X), c=(0,1,0), linestyle='--', label='Poly-fit')
    # axes.plot(X, predict(X, sci_fit), c=(0,0,1), linestyle='-.', label='Gradient')
    # axes.plot(X, predict(X, ours), c='k', linestyle='--', label='Ours')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def example_syn(k=2):
    n,r = pow(2, 12),pow(2, 9)
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'lines.linewidth':5})
    X = np.linspace(1, n, n)
    Y = np.exp(X / r)
    W = np.ones_like(X)
    res,core_res = [], []
    for _ in range(1):
        poly_fit, sci_fit, ours = gen_opt(X,Y,W,k)
        res.append([poly_fit, sci_fit, ours])
        # plot(X, Y, poly_fit, sci_fit, ours)

        ind, C, D, w = coreset_sample(X, Y, pow(2,7), np.abs(predict(X, ours) - Y), predict(X, ours))
        s = np.size(ind) * 3
        print(n/s)
        W_ = np.concatenate([W, w, -w])
        X_ = np.concatenate([X, X[ind], X[ind]])
        Y_ = np.concatenate([Y, C, D])

        poly_fit, sci_fit, ours = gen_opt(X_,Y_,W_,k)
        core_res.append([poly_fit, sci_fit, ours])
        # plot(X, Y, poly_fit, sci_fit, ours)

    res,core_res = np.array(res,dtype=object),np.array(core_res,dtype=object)
    plot(X, Y, res[:,0], res[:,1], res[:,2])
    plot(X, Y, core_res[:, 0], core_res[:, 1], core_res[:, 2])



if __name__ == '__main__':
    example_syn()