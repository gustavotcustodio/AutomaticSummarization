import pandas as pd
import random as rd
import numpy as np
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.metrics import silhouette_score
import cluster
#import datetime

def weighted_euclidean (weights):
    """
    Compute the weighted euclidean distance.

    Parameters
    ----------
    weights: 1d array [float]
        Weights of features.

    Returns
    -------
    ( 1d array [float], 1d array [float] ) -> float
        Distance function.
    """
    def calc_distance (point1, point2):
        
        return np.sqrt(sum(weights*(point1-point2)**2))
   
    return calc_distance


class Evaluator_clustering:
    """
    The class wraps the k-means algorithm, the objective function and the dataset.

    Attributes
    ----------
    data : 2d array [float]
        Text's sentences in bag-of-words format.
    highlights: 2d array [float]
        Research highlights in bag-of-words format.
    weights: 1d array [float]
        Weights for attributes
    indexes_highl: 1d array [int]
        List of sentences indices that are highlights.
    n_clusters: int
        Number of clusters for the clustering algorithm.
    alpha: float
        Weighting factor for the objective function.
    lbound: float
        Min value for each dimension of a particle.
    ubound: float
        Max value for each dimension of a particle.
    labels_results: 1d array [string]
        Labels for the Dataframe containing the results of experiments.
    df_results: DataFrame
        Dataframe storing the fitness function's values in experiments.
    df_weights: DataFrame
        Dataframe storing the weights found in experiments.
    algorithm: string
        Clustering algorithm applied.
    """
    
    def __init__ (self, data, highlights, weights, indexes_highl, n_clusters, 
                    alpha, lbound, ubound, algorithm, sim_matrix = None):
        self.__data             = data
        self.__highlights       = highlights
        self.__indexes_highl    = indexes_highl
        self.__n_clusters       = n_clusters
        self.__alpha            = alpha
        self.__lbound           = lbound
        self.__ubound           = ubound

        self.__labels_results   = ['alpha', 'silhouettes', 'distances', 'evals', 
                                    'n_clusters', 'max_dists_centro_highl']
        self.__df_results       = pd.DataFrame (columns = self.__labels_results)

        self.__labels_weights   = ['alpha'] + [str(i+1) + ' weights' 
                                                for i in range (weights.shape[0])]
        self.__df_weights       = pd.DataFrame (columns = self.__labels_weights)

        self.__algorithm        = algorithm
        self.__rnd_seed         = rd.randint (0, 100)
        self.__define_clusterer (sim_matrix)


    #def _init_weights_labels (self):
        
    #    n_labels_weights = len (filter (lambda d: 'tfidf' not in d, 
    #                                self.__df_results.columns))

    #    labels = ['alpha'] + [str(i+1) + ' weights' 
    #                            for i in range (n_labels_weights)]
    #    print self.__df_results.columns
    #    print labels

    #    return labels

    def __define_clusterer (self, sim_matrix=None):
        if self.__algorithm == 'kmeans':

            self.__clusterer = KMeansClusterer (self.__n_clusters, distance = None, rng=None,
                                                 avoid_empty_clusters=True,  repeats=1 )

        elif self.__algorithm == 'pfcm':
            learning = 0.08
            delta = 0.1
            epsilon =  0.2

            self.__clusterer = cluster.Pfcm (delta, epsilon, learning, self.__data, sim_matrix=sim_matrix)


    def update_alpha (self, alpha):
        """
        Update the alpha and the number of clusters applied in the fitness function.

        Parameters
        ----------
        alpha: float
            Weighting factor to adjust the terms of the fitness function.

        n_clusters: int
            Number of clusters.
        """
        self.__alpha = alpha


    def save_results_and_weights(self, file_name):
        """
        Save the results from the weighted k-means execution to csv files.
    
        Parameters
        ----------
        file_name: str
            Name of file.
        """
        self.__df_results.to_csv (file_name, 
            sep = '\t', encoding = 'utf-8', index = False)
        
        self.__df_weights.to_csv (file_name.replace('_evals_', '_weights_'),                  
            sep = '\t', encoding = 'utf-8', index = False)


    def get_col_results (self, col_name):
        """
        Get a column from the Dataframe containing experiment's results.

        Returns
        -------
        columns: 1d array
            Array containing the values of either columns: 
            alpha, silhouettes, distances, evals, n_clusters, max_dists_centro_highl.

        """
        return self.__df_results [col_name]


    def calc_dists_centro_highl (self, centroids, clusters_highl, norm_weights):
        """
        Compute the average distance between centroids and highlights.

        Parameters
        ----------
        centroids: 1d array [float]
            Centroid values for all clusters.
        clusters_highl: 1d array [int]
            Contains the cluster in which each highlight belongs. 
            The array's indices correspond to the highlights' indices.
        norm_weights: 1d array [float]
            Weights for features normalized between 0 and 1.

        Returns
        -------
        dist_centro_highl: list [(1d array [float], int)]
            List with tuples that contain the distances for all highlights
            in that cluster and the cluster index.

        """
        return [(np.sqrt(sum(w*(centroids[c]-highlight)**2)) , c)
                    for c, highlight, w in zip (clusters_highl, self.__highlights, norm_weights)]


    def calc_max_radius_centro_highl (self, dists_centro_highl):
        """
        Compute the distances for the furthest highlights in each cluster.

        Parameters
        ----------
        dists_centro_highl: list [(array_like[float], int)]
            List with tuples that contain a cluster index and the distances for all 
            highlights in that cluster.

        Returns
        -------
        radius_indices: list [(float, int)]
            Distance of the furthest highlight in that cluster and the cluster index.
        """
        clusters  = np.unique (zip (*dists_centro_highl)[1])
        groupby = [filter (lambda p: p[1]==c, dists_centro_highl) for c in clusters]

        return [(max(zip(*g)[0]), g[0][1]) for g in groupby]


    def sum_dists_centro_highl (self, dist_centro_highl):
        """
        Sum all the distances between centroids and highlights.

		Parameters
        ----------
        dist_centro_highl: list [(1d array [float], int)]
            List with tuples that contain the distances for all highlights
            in that cluster and the cluster index.
			
		Returns
		-------
		sum_dist_centro_highl: float
			Sum of distances between highlights and centroids.
		"""
        
        return sum (zip(*dist_centro_highl)[0])
    

    def append_solution (self, weights):
        """
        Stores 'silhouette', 'weights', 'distances between highlights and centroids'
		and 'objective function value' for the current iteration.
		
		Parameters
        ----------
		weights: 1d array [float]
            Weights for features.
		"""
		
        ev, silh, avg_dists, radius, norm_weights = self.run_weighted_clustering (
            weights, save_params=True)

        self.__df_results = self.__df_results.append ( pd.DataFrame ( 
            np.array([[ self.__alpha, silh, avg_dists, ev, self.__n_clusters, str(radius) ]]), 
                        columns = self.__labels_results ) )

        alpha_and_weights = np.array([np.concatenate([[self.__alpha], norm_weights])])

        self.__df_weights = self.__df_weights.append ( pd.DataFrame ( 
            alpha_and_weights, columns = self.__labels_weights ) )


    def normalize_weights (self, weights):
        """
        Normalize weights between 0 and 1, having the sum always being 1.
        
        Parameters
        ----------
        weights: 1d array [float]
            Weights for features.
            
        Returns
        -------
        norm_weights: 1d array [float]
            Weights normalized between 0 and 1.
        """
        norm_weights = (weights - self.__lbound) / (self.__ubound - self.__lbound)
        return norm_weights / sum (norm_weights)


    def count_clusters_no_highl (self, clusters_highl):     
        """
        Count the number of clusters not containing highlights.
        
        Parameters
        ----------
        clusters_highl: 1d array [int]
            The cluster to which each highlight belongs.
            
        Returns
        -------
        n_clusters_contain_highl: int
            Number of clusters not containing highlights:  
        """ 
        n_clusters_no_highl = 0

        for i in range (self.__n_clusters):
            if i not in clusters_highl:
               n_clusters_no_highl += 1 

        return n_clusters_no_highl
    

    def run_weighted_clustering (self, weights, save_params = False):
        """
        Run the Weighted clustering algorithm.

        Parameters
        ----------
        weights: 1d array [float]
            Weights of features.
        save_params: boolean
            Define if the current solution wii be added to the DataFrame.

        Returns
        -------
        evaluation: float
            Fitness value.
        silhouette: float
            Average silhouette value for all data points.
        avg_dist_centro_highl: float
            Average distance between centroids and highlights.
        radius_centro_highl: 1d array [(float,int)]
            Array containing distances of furthest highlights in each cluster and the cluster index.
        norm_weights: 1d array [float]
            Weights normalized between 0 and 1.
            The sum of these weights is 1.
        """
        # Weights normalized between lbound and ubound
        weights_experiments = self.normalize_weights (weights)
        
        if self.__algorithm == 'kmeans':
            # Run K-Means using a weighted euclidean distance
            rng=rd.Random()
            rng.seed(self.__rnd_seed)

            self.__clusterer._rng = rng
            self.__clusterer._distance =  weighted_euclidean (weights_experiments)
            
            assigned_clusters = self.__clusterer.cluster (self.__data, assign_clusters=True)

            centroids = np.array (self.__clusterer.means())
			
        elif self.__algorithm == 'pfcm':
			# Run then P-FCM algorithm with weights
            self.__clusterer.set_randomizer (self.__rnd_seed)
            u, centroids = self.__clusterer.run_pfmc()
	
            assigned_clusters = np.argmax (u, axis=0)
		
        else:
            # Run Fuzzy-C-Means using a weighted euclidean distance
            n_data = self.__data.shape[0]
            u_starter = np.random.rand (self.__n_clusters, n_data)
            u_starter = cluster._cmeans.normalize_columns (u_starter)

            centroids, u, u0, d, jm, p, fpc = cluster.cmeans(self.__data.T, self.__n_clusters, 
                        2, error=0.005, maxiter=500, init=u_starter, weights= weights_experiments)

            assigned_clusters = np.argmax (u, axis=0)

        # Clusters whose the highlights are associated
        clusters_highl = assigned_clusters [
                            self.__indexes_highl[0]:self.__indexes_highl[-1]+1]

        # Number of clusters that do not contain any highlights
        #n_clusters_no_highl = self.count_clusters_no_highl (clusters_highl)
        
        # All distances between centroids and highlights
        dists_centro_highl = self.calc_dists_centro_highl (
                                centroids, clusters_highl, weights_experiments) 
 
        silhouette = (silhouette_score (self.__data, assigned_clusters, 
                        metric=weighted_euclidean (weights_experiments)) + 1.0) / 2.0
        
        # Average distance between clusters and highlights
        avg_dist_centro_highl = self.sum_dists_centro_highl (
                                dists_centro_highl) / self.__highlights.shape[0]
        
        evaluation = self.__alpha * silhouette - (1-self.__alpha) * avg_dist_centro_highl                                              
        
        if save_params:
            radius_centro_highl = self.calc_max_radius_centro_highl (dists_centro_highl)
            return evaluation, silhouette, avg_dist_centro_highl, radius_centro_highl, weights_experiments
        else:                                
            return evaluation
