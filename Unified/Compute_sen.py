import numpy as np
import Utils
import Coreset
import PointSet


def compute_sen(P_):
    var_dict = Utils.initializaVariables('lz', 1, 1)
    coreset = Coreset.Coreset(P=None, W=None, _sense_bound_lambda=var_dict['SENSE_BOUND'],
                                     max_ellipsoid_iters=var_dict['ELLIPSOID_MAX_ITER'],
                                     problem_type=var_dict['PROBLEM_TYPE'], use_svd=var_dict['USE_SVD'])

    P = PointSet.PointSet(P=P_, ellipsoid_max_iters=100, problem_type='lz',use_svd=True)

    sensitivity = coreset.computeSensitivity(P, P.W)

    return sensitivity
########################################################################################################################
# Copied from Algorithm.py, done to avoid inheritance problems, since the above function is used in Algorithms.
########################################################################################################################
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


def compute_sen_rat(P,k):
    var_dict = Utils.initializaVariables('lz', 1, 1)
    coreset = Coreset.Coreset(P=None, W=None, _sense_bound_lambda=var_dict['SENSE_BOUND'],
                                     max_ellipsoid_iters=var_dict['ELLIPSOID_MAX_ITER'],
                                     problem_type=var_dict['PROBLEM_TYPE'], use_svd=var_dict['USE_SVD'])

    P_ = PointSet.PointSet(P=P, ellipsoid_max_iters=1000,use_svd=False)
    P_.cost_func = lambda x: np.abs(predict(P[:,0],(x[:k],x[k:]))-P[:,1]).sum()
    sensitivity = coreset.computeSensitivity(P_, P_.W)

    return sensitivity

# Test code
if __name__ == '__main__':
    n, d = pow(2, 12), 1
    #   I = np.ones(d)
    #   P = np.linspace(-10*I, 10*I, n)
    #   s = compute_sen(P)
    #   print(np.abs(s.mean() - s).max())
    X = np.linspace(-10, 10, n)
    Y = predict(X,[np.ones(1),np.ones(1)])
    P = np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],1)
    s = compute_sen_rat(P,d)

    print(s)

