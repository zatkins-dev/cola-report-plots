import os
import time
import pandas as pd
import numpy as np
from numpy.linalg import norm
# import torch
# import torch.distributed as dist
from .cocoasolvers import CoCoASubproblemSolver
import cola.communication as comm


class Monitor(object):
    """ Supervising the training process. 

    This class is used to:
    * log the metrics during training (time, loss, etc);
    * save weight file and log files if specified;
    """

    def __init__(self, output_dir, ckpt_freq=-1, exit_time=None, split_by='features', mode='local', Ak=None, Ak_test=None, y_test=None, verbose=1, name=''):
        """
        Parameters
        ----------
        solver : CoCoASubproblemSolver
            a solver to be monitored.
        output_dir : str
            directory of output.
        ckpt_freq : Int
            frequency of the checkpoint.
        exit_time : float, optional
            exit if the program has been running for `exit_time`. (the default is None, which disable this criterion.)
        split_by : str, optional
            The data matrix is split by samples or features (the default is 'samples')
        mode : ['local', 'global', None], optional
             * `local` mode only logs duality gap of local solver. 
             * `global` mode logs duality gap of the whole program. It takes more time to compute.
        """
        self.name = name
        self.Ak = Ak
        self.Ak_test = Ak_test
        self.y_test = y_test
        self.do_prediction_tests = self.Ak_test is not None and self.y_test is not None

        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        self.running_time = 0
        self.previous_time = time.time()
        self.exit_time = exit_time or np.inf

        self.verbose = verbose

        self.records = []
        self.records_l = []
        self.records_g = []
        self.mode = mode
        self.ckpt_freq = ckpt_freq
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None

        # If a problem is split by samples, then the total number of data points is unknown
        # in a local node. As a result, we will defer the division to the logging time.
        self.split_by_samples = split_by == 'samples'

        self._sigma_sum = None

    def init(self, model, graph):
        self.records = []
        self.records_l = []
        self.records_g = []
        self.model = model
        self.solver = self.model.localsolver
        self.beta = graph.beta
       
    def log(self, vk, Akxk, xk, i_iter, solver, delta_xk=None, delta_vk=None, intercept=0.0, cert_cv=0.0):
        # Skip the time for logging
        self.running_time += time.time() - self.previous_time

        if self.mode == 'local':
            self._log_local(vk, Akxk, xk, i_iter, solver, delta_xk, delta_vk, cert_cv=cert_cv)
        elif self.mode == 'global':
            self._log_global(vk, Akxk, xk, i_iter, solver)
        elif self.mode == None:
            pass
        elif self.mode == 'all':
            self.records = self.records_l
            self._log_local(vk, Akxk, xk, i_iter, solver, delta_xk, delta_vk, cert_cv=cert_cv)
            self.records_l = self.records
            self.records = self.records_g
            self._log_global(vk, Akxk, xk, i_iter, solver, intercept=intercept)
            self.records_g = self.records
            if self.verbose >= 2:
                print(f"[{comm.get_rank()}] Certificate, Iter {self.records[-1]['i_iter']}: "
                    f"global_gap={self.records[-1]['gap']:10.5e}; "
                    f"local_gap={self.records_l[-1]['cert_gap']:10.5e}, local_cv={self.records_l[-1]['cv2']:10.5e}")
        else:
            raise NotImplementedError("[local, global, all, None] are expected mode, got {}".format(self.mode))

        self.previous_time = time.time()

        max_running_time = comm.all_reduce(self.running_time, op='MAX')
        gap = 100
        if self.mode == 'all':
            gap = self.records_g[-1]['gap']
        return max_running_time > self.exit_time or abs(gap) < 1e-6

    def _log_local(self, vk, Akxk, xk, i_iter, solver, delta_xk=None, delta_vk=None, cert_cv=0.0):
        record = {}
        record['i_iter'] = i_iter
        record['time'] = self.running_time
        try:
            record['local_gap'] = self.solver.solver.gap_
            record['n_iter_'] = self.solver.solver.n_iter_
        except:
            record['local_gap'] = 0
            record['n_iter_'] = 0

        if delta_vk is not None:
            vklast = vk - self.model.gamma * self.world_size * delta_vk
            record['fk'] = self.solver.f(vklast) / self.world_size
            Pk = self.solver.grad_f(vklast) @ delta_vk
            cvk = self.world_size * np.linalg.norm(delta_vk, 2)**2 / (2 * self.solver.tau)
            record['gk'] = self.solver.gk(xk)
            record['subproblem'] = record['fk'] + Pk + cvk + record['gk']
            record['delta_vk'] = norm(delta_vk, 2)
        else:
            record['fk'] = record['gk'] = record['subproblem'] = record['delta_vk'] = np.nan

        record['delta_xk'] = norm(delta_xk) if delta_xk is not None else np.nan

        self.records.append(record)

        if self.verbose >= 2:
            print("Iter {i_iter:5}, Time {time:10.5e}: delta_xk={delta_xk:10.5e}, local_gap={local_gap:10.5e}, local_iters {n_iter_}".format(**record))

    def _log_global(self, vk, Akxk, xk, i_iter, solver, intercept=0.0):
        record = {}
        record['i_iter'] = i_iter
        record['time'] = self.running_time

        # v := A x
        v = comm.all_reduce(np.array(Akxk), op='SUM')
        w = self.solver.grad_f(v)

        record['res'] = norm(v - self.solver.y) / norm(self.solver.y)

        # Compute squared norm of consensus violation
        record['cv2'] = norm(vk - v, 2) ** 2
        if self.mode == 'all':
            self.records_l[-1]['cv2'] = record['cv2']

        # Compute the value of minimizer objective
        val_gk = self.solver.gk(xk)
        record['g'] = comm.all_reduce(val_gk, 'SUM')
        record['f'] = self.solver.f(v)

        # Compute the value of conjugate objective
        val_gk_conj = self.solver.gk_conj(w)
        record['f_conj'] = self.solver.f_conj(w)
        record['g_conj'] = comm.all_reduce(val_gk_conj, op='SUM')

        if self.split_by_samples:
            n_samples = comm.all_reduce(len(solver.y), op='SUM')
        else:
            n_samples = len(solver.y)

        record['g'] /= n_samples
        record['g_conj'] /= n_samples
        record['f'] /= n_samples
        record['f_conj'] /= n_samples

        # The primal should be monotonically decreasing
        record['P'] = record['f'] + record['g']
        record['D'] = record['f_conj'] + record['g_conj']

        # Duality gap of the global problem
        record['gap'] = (record['D'] + record['P'])

        if self.do_prediction_tests:
            y_predict = self.model.predict(self.Ak_test)
            y_test_avg = np.average(self.y_test)
            record['n_train'] = self.solver.y.shape[0]
            record['n_test'] = self.y_test.shape[0]
            record['rmse'] = np.sqrt(np.average((y_predict - self.y_test)**2))
            record['r2'] = 1.0 - np.sum((y_predict - self.y_test)**2)/np.sum((self.y_test - y_test_avg)**2)
            record['max_rel'] = np.amax(np.abs(y_predict - self.y_test)/self.y_test)
            record['l1_rel'] = np.linalg.norm(self.y_test-y_predict, 1)/np.linalg.norm(self.y_test, 1)
            record['l2_rel'] = np.linalg.norm(self.y_test-y_predict, 2)/np.linalg.norm(self.y_test, 2)

        self.records.append(record)

        if self.rank == 0:
            if self.verbose >= 2:
                print("Iter {i_iter:5}, Time {time:10.5e}: gap={gap:10.3e}, P={P:10.3e}, D={D:10.3e}, f={f:10.3e}, "
                  "g={g:10.3e}, f_conj={f_conj:10.3e}, g_conj={g_conj:10.3e}".format(**record))
                

    def save(self, Akxk=None, intercept=None, modelname=None, logname=None):
        rank = self.rank
        if logname:
            if self.mode == 'all':
                self.records = self.records_g
                logfile = os.path.join(self.output_dir, f'{rank}'+logname)
                pd.DataFrame(self.records_l).to_csv(logfile)
                if self.verbose >= 2:
                    print("Data has been save to {} on node {}".format(logfile, rank))
            if rank == 0:
                logfile = os.path.join(self.output_dir, logname)
                pd.DataFrame(self.records).to_csv(logfile)
                if self.verbose >= 2:
                    print("Data has been save to {} on node 0".format(logfile))

        if modelname:
            modelfile = os.path.join(self.output_dir, modelname)
            if self.split_by_samples:
                Akxk = comm.reduce(Akxk, root=0, op='SUM')
                weight = Akxk
                if rank == 0:
                    import copy
                    import pickle
                    dump_model = copy.deepcopy(self.model)
                    dump_model.coef_ = weight
                    dump_model.intercept_ = self.model.b_offset_ - intercept
                    with open(modelfile, 'wb') as model_file:
                        pickle.dump(dump_model, model_file)
            else:
                self.model.dump(modelfile)
            
            if self.rank == 0 and self.verbose >= 2:
                print("Model has been save to {} on node 0".format(modelfile))
                

    def show_test_statistics(self, n_train=None, intercept=0, Ak_test=None, y_test=None):
        comm.barrier()
        if Ak_test is None:
            Ak_test = self.Ak_test
        if y_test is None:
            y_test = self.y_test
        if Ak_test is None or y_test is None:
            raise TypeError('Ak_test and y_test must not be None')
        
        if n_train is None:
            n_train = self.Ak.shape[0]
        n_test = len(y_test)
        
        if self.mode in ['global', 'all']:
            y_predict = self.model.predict(self.Ak_test)
            y_test_avg = np.average(y_test)
            rmse = np.sqrt(np.average((y_predict - y_test)**2))
            r2 = 1.0 - np.sum((y_predict - y_test)**2)/np.sum((y_test - y_test_avg)**2)
            max_rel = np.amax(np.abs(y_predict - y_test)/y_test)
            l1_rel = np.linalg.norm(y_test-y_predict, 1)/np.linalg.norm(y_test, 1)
            l2_rel = np.linalg.norm(y_test-y_predict, 2)/np.linalg.norm(y_test, 2)

        if self.verbose >= 1 and comm.get_rank() == 0:
            print(f'|-> Test Statistics ({n_train}/{n_test}/{n_train + n_test}): ')
            print(f'|---> max. rel. error = {max_rel}')
            print(f'|--->   rel. L1 error = {l1_rel}')
            print(f'|--->   rel. L2 error = {l2_rel}')
            print(f'|--->            RMSE = {rmse}')
            print(f'|--->             R^2 = {r2}')
