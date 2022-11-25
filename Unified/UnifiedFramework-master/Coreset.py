import numpy as np
import PointSet
# from MainProgram.Utils import SENSE_BOUND
# from MainProgram import Utils
import time
import copy


class Coreset(PointSet.PointSet):
    def __init__(self, P=None, W=None, _sense_bound_lambda=None,
                 max_ellipsoid_iters=10, use_svd=True, problem_type=None):
        PointSet.PointSet.__init__(self,P=P, W=W, ellipsoid_max_iters=max_ellipsoid_iters, problem_type=problem_type,
                                   use_svd=use_svd)
        self.rank = 0
        self.sensitivities = None
        self._sense_bound_lambda = _sense_bound_lambda
        self.probability = None
        self.S = None
        self.rank = 0
        self.use_svd = use_svd
        self.time_taken = 0
        self.problem_type = problem_type

    def sampleCoreset(self, P, sensitivity, sample_size, random_state=0):
        startTime = time.time()
        weights = P.W
        # Compute the sum of sensitivities.
        t = np.sum(sensitivity)

        # The probability of a point prob(p_i) = s(p_i) / t
        self.probability = sensitivity.flatten() / t

        # The number of points is equivalent to the number of rows in P.
        n = P.n

        # initialize new seed
        np.random.seed(random_state)

        # Multinomial distribution.
        indxs = np.random.choice(n, sample_size, p=self.probability.flatten())

        # Compute the frequencies of each sampled item.
        hist = np.histogram(indxs, bins=range(n))[0].flatten()
        indxs = copy.deepcopy(np.nonzero(hist)[0])

        # Select the indices.
        S = P.P[indxs, :]

        # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
        weights = np.asarray(np.multiply(weights[indxs], hist[indxs]), dtype=float).flatten()

        # Compute the weights of the coreset
        weights = np.multiply(weights, 1.0 / (self.probability[indxs] * sample_size))
        self.time_taken = time.time() - startTime

        self.S = PointSet.PointSet(P=S, W=weights, ellipsoid_max_iters=self.ellipsoid_max_iters,
                                   problem_type=self.problem_type, use_svd=self.use_svd, compute_U=False)
        return self.S, self.time_taken

    def computeSensitivity(self, P, sum_old_weights=None):
        sensitivitiy = np.empty(P.n, )

        if sum_old_weights is None:
            sum_old_weights = np.sum(P.W)

        sensitivity = np.empty((P.n, ))
        if 'lz' in self.problem_type:
            sensitivity = self._sense_bound_lambda(P.P, P.W, P.d)
        else:
            sensitivity[P.pos_idxs] = self._sense_bound_lambda(x=P.U[P.pos_idxs, :], w=P.W[P.pos_idxs],
                                                                args=(P.sum_weights_pos, P.sum_weights_neg,
                                                                      P.sum_weights)
                                                                )
            sensitivity[P.neg_idxs] = self._sense_bound_lambda(x=P.U[P.neg_idxs, :], w=P.W[P.neg_idxs],
                                                                args=(P.sum_weights_neg, P.sum_weights_pos,
                                                                      P.sum_weights)
                                                                )
        return sensitivity

    def mergeCoresets(self, coreset, sample_size):
        self.S.mergePointSet(coreset.S)
        sens = self.computeSensitivity(P=self.S)
        self.sampleCoreset(P=self.S, sensitivity=sens, sample_size=sample_size)
        self.rank += 1


# SENSE_BOUNDS = {
#     'logisitic': (lambda x, w, args=None: 32 / LAMBDA * (2 * w / args[0] + w * np.linalg.norm(x, 2) ** 2) * args[0]),
#     'nclz': (lambda x, w, args=None: w * np.linalg.norm(x, ord=Z, axis=1) ** Z),
#     'svm': (lambda x, w, args=None: max(9 * w / args[0], 2 * w / args[1]) + 13 * w / (4 * args[0]) +
#                                     125 * (args[2]) / (4 * LAMBDA) * (w * np.linalg.norm(x, 2)**2 +
#                                                                           w/(args[0] + args[1]))),
#     'restricted_lz': (lambda x, w, args=None: w * min(np.linalg.norm(x, ord=2),
#                                                     args[1] ** np.abs(0.5 - 1/Z) * np.linalg.norm(args[0]))),
#     'lz': (lambda x, w, args=None: w * np.linalg.norm(x, Z) ** Z if 1 <= Z <= 2
#             else args[0]**(Z/2) * w * np.linalg.norm(x, ord=Z, axis=1)**Z),
#     'lse': (lambda x, w, args=None: w * np.linalg.norm(x, 1, axis=1))
# }
