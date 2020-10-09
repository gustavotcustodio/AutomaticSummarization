import random as rd
import numpy as np


# PSO algorithm
class Pso:
    """
    PSO algorithm implementation.

    Attributes
    ----------
    fitness: ( 1d array [float] ) -> float
        Evaluation function for particles.
    length_part: int
        Particle's length.
    n_part: int
        Number of particles.
    evaluator: Evaluator_clustering
        Class containing the fitness function and the data.
    max_iter: int
        Max number of iterations for the PSO.
    inertia: float
        PSO constant.
    c1: float
        PSO constant.
    c2: float
        PSO constant.
    ubound: float
        Max value for each dimension of a particle.
    lbound: float
        Min value for each dimension of a particle.
    """

    def __init__(self, fitness, length_part, n_part, evaluator=[],
                 max_iter=100, inertia=0.7298, c1=1.49618, c2=1.49618,
                 ubound=1, lbound=-1):
        self._fitness = fitness
        self._length_part = length_part
        self._n_part = n_part
        self._evaluator = evaluator
        self._inertia = inertia
        self._c1 = c1
        self._c2 = c2
        self._ubound = ubound
        self._lbound = lbound
        self._max_iter = max_iter

    def _get_bound_args(self, x):
        """
        Return the array's indexes with value lower than the lower bound
        and value higher than the upper bound.

        Parameters
        ----------
        x: 1d array [float]
            Position or velocity of particle.

        Returns
        -------
        indices_x_upper: 1d array [int]
            Array containing the indices violating the upper bound.
        indices_x_lower: 1d array [int]
            Array containing the indices violating the lower bound.
        """
        return np.argwhere(x > self._ubound).flatten(), np.argwhere(
            x < self._lbound).flatten()

    def bound_handling(self, pos_part_i, vel_part_i):
        """
        Check if a particle is out of the search space bounds, and fix it.

        Parameters
        ----------
        pos_part_i: 1d array [float]
            Position of particle i.
        vel_part_i: 1d array [float]
            Velocity of particle i.
        """
        coef_vel = 0.5

        ind_ubounds, ind_lbounds = self._get_bound_args(pos_part_i)
        if ind_ubounds.shape[0]:
            pos_part_i[ind_ubounds] = 2*self._ubound - pos_part_i[ind_ubounds]
        if ind_lbounds.shape[0]:
            pos_part_i[ind_lbounds] = 2*self._lbound - pos_part_i[ind_lbounds]

        vel_part_i = -coef_vel * vel_part_i

    def limit_velocity(self, vel_part_i):
        """
        Check if velocity is in the bounds and change the values that are not.

        Parameters
        ----------
        vel_part_i: 1d array [float]
            Velocity of particle i.
        """
        ind_ubounds = np.argwhere(vel_part_i > self._ubound).flatten()
        vel_part_i[ind_ubounds] = self._ubound

        ind_lbounds = np.argwhere(vel_part_i < self._lbound).flatten()
        vel_part_i[ind_lbounds] = self._lbound

    def run_pso(self):
        """
        Run PSO algorithm.

        Returns
        -------
        g: 1d array [float]
            Best solution found by the PSO.
        """
        # Position of particles in the search space
        pos_part = np.array([[rd.random() for _ in range(self._length_part)]
                            for i in range(self._n_part)])
        fitness_pos = np.apply_along_axis(
            self._fitness, 1, pos_part)  # Fitness of particles

        # Velocity of particles
        vel_part = np.array([[rd.uniform(
            -abs(self._ubound-self._lbound), abs(self._ubound-self._lbound))
            for _ in range(self._length_part)] for i in range(self._n_part)])

        pbest = np.copy(pos_part)
        fitness_pbest = np.copy(fitness_pos)

        gbest = np.copy(pbest[np.argmax(fitness_pbest)])
        fitness_gbest = np.max(fitness_pbest)

        # Store the current solution for the current iteration
        self._evaluator.append_solution(gbest)

        print('-------------------- eval: '+str(fitness_gbest) +
              " --------------------")

        n_iter = 0
        while n_iter < self._max_iter:
            for i in range(self._n_part):
                # Update velocity:
                # v = w v + c1 r1 pbest[i] - x[i]) + c2 r2 (gbest - x[i])
                vel_part[i] = self._inertia * vel_part[i] + \
                    self._c1 * rd.random() * (pbest[i] - pos_part[i]) + \
                    self._c2 * rd.random() * (gbest - pos_part[i])

                # Put velocity between lbound and ubound
                self.limit_velocity(vel_part[i])

                # Update position
                pos_part[i] += vel_part[i]

                # Put particle between lbound and ubound
                self.bound_handling(pos_part[i], vel_part[i])

                fitness_pos[i] = self._fitness(pos_part[i])

                if fitness_pos[i] > fitness_pbest[i]:
                    pbest[i] = np.copy(pos_part[i])
                    fitness_pbest[i] = fitness_pos[i]

                    if fitness_pos[i] > fitness_gbest:
                        gbest = np.copy(pos_part[i])
                        fitness_gbest = fitness_pos[i]

            n_iter += 1

            # Store the global solution for this iteration
            self._evaluator.append_solution(gbest)
            print('-------------------- eval: '+str(fitness_gbest) +
                  " --------------------")
        return gbest
