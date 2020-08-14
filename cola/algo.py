r"""CoCoA family to use."""
import warnings
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csc_matrix, issparse

from . import communication as comm


def run_algorithm(algorithm, Ak, b, solver, gamma, theta, max_global_steps, local_iters, graph, monitor, fit_intercept=False, normalize=False):
    r"""Run cocoa family algorithms."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            'Objective did not converge. You might want to increase the number of iterations. '
            'Fitting data with very small alpha may cause precision problems.')

        comm.barrier()
        if algorithm == 'cola':
            model = Cola(gamma, solver, theta, fit_intercept, normalize)
            monitor.init(model)
            model = model.fit(Ak, b, graph, monitor, max_global_steps, local_iters)
        else:
            raise NotImplementedError()
    return model

class Cola:
    def __init__(self, gamma, localsolver, theta, fit_intercept, normalize):
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma should in (0, 1]: got {}".format(gamma))
        self.gamma = gamma
        self.localsolver = localsolver
        self.theta = theta
        self.coef_ = None
        self.intercept_ = None
        self.Ak_offset_ = None
        self.b_offset_ = 0.0
        self.Ak_scale_ = None
        self.fit_intercept = fit_intercept
        self.is_fit = False
        self.coef_scaled = False
        self.normalize = normalize
    
    def fit(self, Ak, b, graph, monitor=None, global_iters=200, local_iters=10):
        Ak, b, self.Ak_offset_, self.b_offset_, self.Ak_scale_ = self._prepare_data(Ak, b)

        # Shape of the matrix
        n_rows, n_cols = Ak.shape

        # Current rank of the node
        rank = comm.get_rank()
        K = comm.get_world_size()

        # Initialize
        self.coef_ = np.zeros(n_cols)
        Akxk = np.zeros(n_rows)

        # Keep a list of neighborhood and their estimates of v
        local_lookups = graph.get_neighborhood(rank)
        local_vs = {node_id: np.zeros(n_rows)
                    for node_id, _ in local_lookups.items()}

        sigma = self.gamma * K
        self.localsolver.dist_init(Ak, b, self.theta, local_iters, sigma)

        # Initial
        comm.p2p_communicate_neighborhood_tensors(
            rank, local_lookups, local_vs)

        if monitor is not None:
            monitor.log(np.zeros(n_rows), Akxk, self.coef_, 0, self.localsolver)
        
        for i_iter in range(1, 1 + global_iters):
            # Average the local estimates of neighborhood and self
            averaged_v = comm.local_average(n_rows, local_lookups, local_vs)
            
            # Solve the suproblem using this estimates
            delta_xk, delta_v = self.localsolver.solve(averaged_v, Akxk, self.coef_)

            # update local variables
            self.coef_ += self.gamma * delta_xk
            Akxk += self.gamma * delta_v

            # update shared variables
            averaged_v += self.gamma * delta_v * K
            local_vs[rank] = averaged_v
            comm.p2p_communicate_neighborhood_tensors(
                rank, local_lookups, local_vs)

            intercept_ = -self.intercept
            if monitor and monitor.log(averaged_v, Ak @ self.coef_, self.coef_, i_iter, self.localsolver, delta_xk=delta_xk, delta_vk=delta_v, intercept=intercept_):
                if monitor.verbose >= 1 and rank == 0:
                    print(f'break @ iter {i_iter}.')
                break

            if monitor and monitor.ckpt_freq > 0 and (i_iter % monitor.ckpt_freq) == 0:
                monitor.save(Akxk, intercept=intercept_, modelname='model_epoch_{}.pickle'.format(i_iter))
        
        self.is_fit = True

        return self

    def predict(self, A):
        if comm.get_world_size() > 0:
            yk = safe_sparse_dot(A, self.coef) + self.intercept
            y = comm.all_reduce(yk, op='SUM')
            return y + self.b_offset_

        elif self.is_fit:
            y = np.resize(A@self.coef + self.intercept, (A.shape[0],1))
            return y

    @property
    def coef(self):
        if self.coef_ is None:
            return None
        if not self.is_fit:
            return self.coef_/self.Ak_scale_
        if not self.coef_scaled:
            self.coef_ = self.coef_/self.Ak_scale_
            self.coef_scaled = True
        return self.coef_

    @property
    def intercept(self):
        if self.fit_intercept:
            if not self.is_fit:
                return -np.dot(self.Ak_offset_, self.coef.T)

            if self.intercept_ is None:
                self.intercept_ = -np.dot(self.Ak_offset_, self.coef.T)
            
            return self.intercept_

        return 0.

    def _prepare_data(self, Ak, b):
        from .dataset import _preprocess_data
        if self.normalize and not self.fit_intercept:
            Ak, b, Ak_offset, b_offset, Ak_scale = _preprocess_data(Ak, b, self.fit_intercept, normalize=self.normalize)
        else:
            Ak, b, Ak_offset, b_offset, Ak_scale = _preprocess_data(Ak.todense(), b, self.fit_intercept, normalize=self.normalize, return_mean=True)
            Ak = csc_matrix(Ak)
        return Ak, b, Ak_offset, b_offset, Ak_scale
    
    def _finalize(self):
        self.is_fit = True

    def dump(self, modelpath):
        size = np.array([0] * comm.get_world_size())
        rank = comm.get_rank()
        size[rank] = len(self.coef)
        size = comm.all_reduce(size, op='SUM')

        weight = np.zeros(sum(size))
        weight[sum(size[:rank]): sum(size[:rank]) + len(self.coef)] = np.array(self.coef)
        weight = comm.reduce(weight, root=0, op='SUM')

        intercept = comm.reduce(self.intercept, root=0, op='SUM')
        if rank == 0:
            import copy
            import pickle
            dump_model = copy.deepcopy(self)
            dump_model.coef_ = weight
            dump_model.intercept_ = self.b_offset_ + intercept
            with open(modelpath, 'wb') as model_file:
                pickle.dump(dump_model, model_file)
