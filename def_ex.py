import numpy as np
import matplotlib.pyplot as plt


def plot():
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.legend()
    plt.show()


def ex1():
    X = np.linspace(-1,1,100_000)
    Y = 1/(1+100*X**2)
    f = np.polynomial.polynomial.Polynomial.fit(X,Y,10).coef
    plt.scatter(X,Y,c='r',label='GT',s=100)
    plt.plot(X, np.polyval(np.flip(f),X), c='k',lw=3,label='Poly-fit deg=10')
    plt.plot(X, 1/(1+100*X**2), c='b',ls='--',lw=3, label='RFF deg =2')
    plot()

def cent_pred(X, c):
    return 1 / (1 + X * c)

def ex2():
    X = np.arange(1,25,1)
    plt.axvline(15, c='k', ls='--', lw=10)
    plt.scatter(X, np.zeros_like(X), c='k', label='P', s=250)
    plt.scatter(15, 0, c='r', label='(a,0)', s=1500)
    R = np.arange(1,25,1e-3)
    Rl,Rr = R[R>15.1],R[R<14.9]
    plt.plot(Rl,cent_pred(Rl, -1/15), c='r', label='query', lw=10)
    plt.plot(Rr, cent_pred(Rr, -1 / 15), c='r', lw=10)
    plot()

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 50})
    ex1()