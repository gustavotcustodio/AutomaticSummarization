import numpy as np
import random
from scipy.spatial import distance_matrix
import _cmeans as cm

AXIS_ROW = 0
AXIS_COL = 1


class Pfcm:
    def __init__(self, delta, epsilon, learning, data, dists=None,
                 prox_hints=None, sim_matrix=None):
        self.delta = delta
        self.epsilon = epsilon
        self.learning = learning
        self.data = data

        if dists is not None:
            self.dists = dists
        else:
            self.calc_data_dists()

        if prox_hints is not None:
            self.prox = prox_hints
        elif sim_matrix is not None:
            self.calc_prox_hints(sim_matrix)

    def set_randomizer(self, random_seed):
        self._random_seed = random_seed

    def calc_data_dists(self):
        self.dists = distance_matrix(self.data, self.data)

    def calc_prox_hints(self, sim_matrix):
        prox = []
        for i in range(sim_matrix.shape[0]):
            indices_hints = np.argwhere(sim_matrix[i] > 0)[0]
            for j in indices_hints:
                prox += [[i, j, sim_matrix[i, j]]]

        self.prox = np.array(prox)

    def _calc_induced_proximity(self, u_i, u_j):
        indices_min = np.where(u_i < u_j)
        return np.sum(np.delete(u_j, indices_min)) + np.sum(u_i[indices_min])

    def _calc_v(self, u):
        v = 0.0
        for p in self.prox:
            i, j, prox_value = int(p[0]), int(p[1]), p[2]
            p_induced = self._calc_induced_proximity(u[:, i], u[:, j])
            v += (p_induced - prox_value)**2 * self.dists[i, j]
        return v

    def _inner_deriv(self, u, s, t, i, j):
        if (t == i and u[s,i] <= u[s,j]) or (t == j and u[s,j] <= u[s,i]):
            return 1
        else:
            return 0


    # Testado
    def _calc_partial_deriv (self, u, s, t):

        sum_proximities = 0

        for p in self.prox:
            i, j, prox_value = int(p[0]), int(p[1]), p[2]

            if self._inner_deriv (u, s, t, i, j):
                induced_prox = self._calc_induced_proximity (u[:,i], u[:,j])
                sum_proximities += (induced_prox - prox_value) #* self.dists[i,j]

        return  2 * sum_proximities


    def _gradient_optimization (self, u):
        v = float("inf")
        v_previous = 0
        n_iters = 0

        # Repeat while the distance is higher than epsilon
        while abs(v - v_previous) > self.epsilon and n_iters < 3:
            v_previous = v

            for s in range(u.shape[0]):
                for t in range(u.shape[1]):

                    partial = self._calc_partial_deriv (u, s, t)

                    u[s,t] = np.clip( u[s,t] - self.learning * partial, 0, 1)

            # Performance index to be minimized
            v = self._calc_v (u)
            n_iters += 1

        sum_clusters = np.sum (u, axis = 0)

        partit_m = u / sum_clusters
        partit_m [np.isnan (partit_m)] = 1.0 / 3.0

        return partit_m


    def _normalize_part_matrix (self, u):
        f = lambda x: x / np.sum(x)
        return np.apply_along_axis (f, AXIS_ROW, u)


    def _get_dist_u (self, u, u_ant):
        if u is None:
            return float('inf')
        return np.linalg.norm(u - u_ant)


    def _calc_centers (self, u):

        return u.dot (self.data) / np.atleast_2d(np.sum(u,axis=1)).T


    def run_pfmc (self):
        n_iters = 0

        m = 2
        n_centers = self.prox.shape[1]

        np.random.seed (seed=self._random_seed)
        u = np.random.rand (n_centers, self.data.shape[0])
        u = self._normalize_part_matrix (u)
        stopping = False

        # Repeat while the distance is higher than delta
        while not(stopping) and n_iters < 4:

            u_ant = np.copy(u)

            # Run 1 iter of FCM
            centers, u, u0, d, jm, p, fpc = cm.cmeans(
                self.data.T, n_centers, m, error=0.0, maxiter=2, init=u)

            # Internal optimization loop
            u = self._gradient_optimization (u)

            stopping = self._get_dist_u (u, u_ant) <= self.delta
            n_iters += 1

        return u, self._calc_centers (u**m)


if __name__ == "__main__":

    n_centers = 3

    data = np.random.rand (100, 8)

    dists = distance_matrix (data, data)

    indices_hints = np.array( range(data.shape[0]) )
    np.random.shuffle (indices_hints)
    indices_hints = indices_hints[:20]

    prox = []

    for i in range (0, indices_hints.shape[0], 2):
        i1 = indices_hints[i]
        i2 = indices_hints[i+1]
        prox_value = random.random()
        prox += [[i1, i2, prox_value]]
        prox += [[i2, i1, prox_value]]

    for i in range(data.shape[0]):
        prox += [[i,i, 1.0]]
    prox = np.array ( prox )

    learning = 0.05
    delta = 0.005
    epsilon =  0.05

    pfcm = Pfcm (data, prox, dists, delta, epsilon, learning)
    print(pfcm.run_pfmc())
