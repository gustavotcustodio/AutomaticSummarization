import os
import math
import pandas as pd
import numpy as np

from swarm import pso_exec as Optim
from evaluator import evaluator_weights_features as Eval


def _get_best_n_clusters(data, highlights, indexes_highl, max_n_clusters,
                         alpha, weights, lbound, ubound, cluster_alg):
    """
    Get the optimal number of clusters according to the silhouette.

    Parameters
    ----------
    data: 2d array [float]
        Text's sentences in bag-of-words format.
    highlights: 2d array [float]
        Research highlights in bag-of-words format.
    indexes_highl: 1d array [int]
        List of sentences indices that are highlights.
    max_n_clusters: int
        Max number of clusters for silhouette score.
    alpha: float
        Weighting factor for the objective function.
    weights: 1d array [float]
        Weights of features.
    lbound: float
        Min value for each weight.
    ubound: float
        Max value for each weight.

    Returns
    -------
    best_n_clusters: int
        Number of clusters with best silhouette score.
    """
    silhouettes_test = []

    # Test all possible number of clusters from 2 to sqrt(num_data_instances)
    for n_clusters in range(2, int(max_n_clusters+1)):

        evaluator = Eval.Evaluator_clustering(
            data, highlights, weights, indexes_highl, n_clusters, alpha,
            lbound, ubound, cluster_alg)

        evaluator.run_weighted_clustering(weights)
        evaluator.append_solution(weights)

        silhouettes_test.append(evaluator.get_col_results('silhouettes')[0])
    # Get the number of cluster with highest average silhouette

    return np.argmax(silhouettes_test) + 2


if __name__ == "__main__":
    folder_name = 'files_cluster_values'

    dir_cluster_values = os.path.join(os.path.dirname(__file__), folder_name)

    files = os.listdir(dir_cluster_values)
    files_value_arrays = [os.path.join(dir_cluster_values, f) for f in files]

    files_value_arrays = sorted(files_value_arrays)

    cluster_alg = 'pfcm'
    # Possible values for weights in the fitness function
    alphas = [1.0/2.0, 2.0/3.0, 1.0/3.0]
    # Read all files in the folder "files_cluster_values"
    for f in files_value_arrays:
        print(f)
        raw_data = pd.read_csv(f, delimiter="\t", encoding='utf-8')

        sim_matrix = np.loadtxt(f.replace(folder_name, 'files_sim_matrices'),
                                delimiter=";", encoding='utf-8')

        cols = filter(lambda c: 'tfidf' not in c, raw_data.columns)

        n_cols = len(cols) - 1

        # Indexes of highlights sentences
        indexes_highl = raw_data.index[raw_data['is_a_highlight'] == 1.0
                                       ].tolist()
        # Normalize the data from 0 to 1
        data = np.apply_along_axis(
            lambda x: (x-min(x)) / (max(x)-min(x))
            if max(x) > 0 else 0, 0, raw_data.values[:, 1:])

        highlights = data[indexes_highl]
        n_clusters = len(highlights)

        # The maximum number of clusters is defined as the square root of the
        # number of data instances
        max_n_clusters = math.floor(np.sqrt(data.shape[0]))

        lbound, ubound = -1.0, 1.0

        for run in range(1, 6):
            weights = np.array([1.0/n_cols for _ in range(n_cols)])

            if cluster_alg == 'pfcm':
                evaluator = Eval.Evaluator_clustering(
                    data, highlights, weights, indexes_highl, n_clusters, 0.5,
                    lbound, ubound, cluster_alg, sim_matrix=sim_matrix)
            else:
                evaluator = Eval.Evaluator_clustering(
                    data, highlights, weights, indexes_highl, n_clusters, 0.5,
                    lbound, ubound, cluster_alg)

            for a in alphas:
                weights = np.array([1.0/n_cols for _ in range(n_cols)])

                # Change the evaluator settings
                evaluator.update_alpha(a)

                fitness_func = evaluator.run_weighted_clustering

                print('******************************************************')
                print('Starting pso...')

                # default: 100 particles 50 iters ....30
                solution = Optim.Pso(fitness_func, n_cols, n_part=100,
                                     max_iter=10, evaluator=evaluator
                                     ).run_pso()
                print('======================================================')

            path = os.path.join(os.path.dirname(__file__), 'files_results')
            basename = os.path.split(f)[-1].replace('.csv', '')
            file_name = '%s_evals_%d.csv' % (basename, run)

            full_file_name = os.path.join(path, file_name)
            evaluator.save_results_and_weights(full_file_name)
            print('Files saved')
