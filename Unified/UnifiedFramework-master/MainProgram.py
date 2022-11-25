import numpy as np
from scipy import stats
import Utils
import Optimizor
from multiprocessing.pool import ThreadPool
import Coreset
import copy


class MainProgram(object):
    def __init__(self, file_path, problem_type, Z, LAMBDA=1, streaming=False):
        self.file_path = file_path
        self.file_name = self.file_path.split('.')
        Utils.checkIfFileExists(file_path)
        self.pool = ThreadPool(Utils.THREAD_NUM)
        var_dict = Utils.initializaVariables(problem_type, Z, LAMBDA)
        self.coresets = [Coreset.Coreset(P=None, W=None, _sense_bound_lambda=var_dict['SENSE_BOUND'],
                                         max_ellipsoid_iters=var_dict['ELLIPSOID_MAX_ITER'],
                                         problem_type=var_dict['PROBLEM_TYPE'], use_svd=var_dict['USE_SVD'])
                         for i in range(Utils.REPS)]
        self.samplingProcedures = \
            (lambda i, sensitivity, sample_size, random_state=None:
             self.coresets[i].sampleCoreset(P=self.P, sensitivity=sensitivity,
                                            sample_size=sample_size, random_state=random_state)) if not streaming else 1

        self.P = Utils.readRealData(file_path)
        self.sample_sizes = Utils.generateSampleSizes(self.P.n)
        self.optimizor = Optimizor.Optimizor(P=self.P, Z=var_dict['Z'], C=var_dict['LAMBDA'],
                                             problem_type=var_dict['PROBLEM_TYPE'], tol=var_dict['OPTIMIZATION_TOL'],
                                             objective_cost=var_dict['OBJECTIVE_COST'])
        self.optimizor.defineSumOfWegiths(self.P.W)
        if not streaming:
            self.sensitivity = self.coresets[0].computeSensitivity(self.P, self.P.sum_weights)

        self.opt_val, self.opt_time = self.optimizor.fit(self.P)

    def computeRelativeErrorPerCoreset(self, coreset, time_taken):
        value_on_coreset, fitting_time = self.optimizor.fit(coreset)
        return value_on_coreset / self.opt_val - 1, fitting_time + time_taken

    @classmethod
    def updateProblem(cls, file_path, problem_type, Z, LAMBDA=1, streaming=False):
        cls.pool.close()
        cls.pool.join()
        cls.pool = ThreadPool(Utils.THREAD_NUM)
        cls.file_path = file_path
        Utils.initializaVariables(problem_type, Z, LAMBDA)
        cls.P = Utils.readRealData(cls.file_path)
        cls.samplingProcedures = \
            (lambda i, sensitivity, sample_size, random_state=0, is_uniform=False:
             cls.coresets[i].sampleCoreset(sensitivity, sample_size, random_state)) \
                if not streaming else 1

    def computeAverageEpsAndDelta(self, sensitivity, sample_size):
        func = lambda x: self.computeRelativeErrorPerCoreset(x[0], x[1])

        # self.samplingProcedures(i=0, sensitivity=sensitivity, sample_size=sample_size)
        if Utils.PARALLELIZE:  # if parallel computation is enabled
            coresets = self.pool.map(lambda i: self.samplingProcedures(i, sensitivity, sample_size), range(Utils.REPS))
            relative_errors_and_time = self.pool.map(func, coresets)
        else:  # otherwise
            coresets = [self.samplingProcedures(i, sensitivity, sample_size) for i in range(Utils.REPS)]
            relative_errors_and_time = [func(x) for x in coresets]

        # return mean of error,
        #        mean of time,
        #        median absolute deviation of errors,
        #        median absolute deviation of times,
        #        mean coreset size
        return np.mean([x[0] for x in relative_errors_and_time]), \
               np.mean([x[1] for x in relative_errors_and_time]), \
               stats.median_absolute_deviation(np.array([x[0] for x in relative_errors_and_time])), \
               stats.median_absolute_deviation(np.array([x[0] for x in relative_errors_and_time])), \
               np.mean([x[0].d for x in coresets])

    def applyComaprison(self):
        mean_of_error = np.empty((Utils.NUM_SAMPLES, 2))
        mean_of_time = copy.deepcopy(mean_of_error)
        std_of_error = copy.deepcopy(mean_of_error)
        std_of_time = copy.deepcopy(mean_of_error)
        mean_of_coreset_size = copy.deepcopy(mean_of_error)
        all_sensitivities = np.vstack((np.ones(self.sensitivity.shape), self.sensitivity))

        for i in range(Utils.NUM_SAMPLES):
            sample_size = self.sample_sizes[i]
            for type_of_sampling_alg in range(2):
                mean_of_error[i, type_of_sampling_alg], mean_of_time[i, type_of_sampling_alg],\
                    std_of_error[i, type_of_sampling_alg], std_of_time[i, type_of_sampling_alg], \
                    mean_of_time[i, type_of_sampling_alg] = \
                    self.computeAverageEpsAndDelta(all_sensitivities[type_of_sampling_alg,:],
                                                   int(sample_size))

        np.savez(r'results/{}/results.npz'.format(self.file_name), mean_of_error=mean_of_error,
                 mean_of_time=mean_of_time, std_of_error=std_of_error,
                 std_of_time=std_of_time, mean_of_coreset_size=mean_of_coreset_size, coreset_size=self.sample_sizes)


    @staticmethod
    def main():
        dataset = 'HTRU_2.csv'
        problem_type = 'svm'
        Z = 2
        Lambda = 1
        streaming = False
        main_runner = MainProgram(dataset, problem_type, Z, Lambda, streaming)
        main_runner.applyComaprison()



if __name__ == '__main__':
    MainProgram.main()