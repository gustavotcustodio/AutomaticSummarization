import numpy as np 
import pandas as pd
import sys
import os 
import io
import tokenizer
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.cluster import KMeans
from cluster import Pfcm
from scipy.spatial import distance
from evaluator import evaluator_weights_features as ev


def cluster_sentences (sents_vecs, num_clusters, weights=None):
    if weights is not None:
        print 'com pesos'
        dist_metric = ev.weighted_euclidean (weights)
    else:
        print 'sem pesos'
        dist_metric = distance.euclidean

    kclusterer = KMeansClusterer (num_clusters, repeats=1, distance=dist_metric,
                                    avoid_empty_clusters=True) 

    labels = kclusterer.cluster (sents_vecs, assign_clusters=True)
            
    centroids = np.array (kclusterer.means())

    return np.array(labels), centroids


def rank_sentences (data, labels, centroids):
    '''
    Returns
    -------
    ranked_sentences: dict (int -> 2d array)
        For each cluster returns a ranking containing the
        closest sentences to the centroid of that cluster.
    '''    
    n_clusters = centroids.shape[0]

    ranked_sentences = {}

    for c in range(n_clusters):

        indices = np.argwhere(labels==c).flatten()
        dists = [np.linalg.norm (data[i]-centroids[c]) for i in indices]
    
        index_dists = zip (indices, dists)

        ranked_sentences[c] = sorted (index_dists, key=lambda tup: tup[1])
    return ranked_sentences


def save_ranking_file (ranking, sentences, ranked_highlights, file_name):
    file_to_write = io.open (file_name, mode='w', encoding='utf-8')

    for r in ranking:
        file_to_write.write (u'\n---------- Cluster ' + str(r) + ' ----------\n')

        for item in ranking[r]:
            sent = sentences[item[0]].replace('\n', ' ').encode('utf-8')
            row = '{:<5}=>{}\n'.format (item[0], sent)
            file_to_write.write (row.decode('utf-8'))   

    file_to_write.write (u'\n-------------------------------\n')
    file_to_write.write (u'Cluster    Rank pos    Sentence\n')
    file_to_write.write (  u'-------------------------------\n')

    for tup in ranked_highlights:
        sent = sentences[tup[2]].replace('\n', ' ').encode('utf-8')
        row = '{:<11}{:<12}{}\n'.format (tup[0], tup[1]+1, sent)
        file_to_write.write (row.decode('utf-8'))

    file_to_write.close()


def mark_highlights (sentences, n_highlights, file_name):
    file_name = file_name.replace('files_txt', 'files_highlights')
    n_highlights = len(tokenizer.get_highlights(file_name))
    n_sentences = len(sentences)

    for i in range(n_sentences-n_highlights, n_sentences):
        sentences[i] = '(HIGHLIGHT) ' + sentences[i]


def define_weights (n_attr):
    w_abstract = 0.5
    w_others = (1.0 - w_abstract) / (n_attr-1)

    weights = [w_abstract]
    weights += [w_others for _ in range(n_attr-1)]
    return np.array (weights)


def get_ranking_highlights (sentences, ranked_sents, n_highl):
    n_sents = len(sentences)
    indices_highl = range(n_sents-n_highl, n_sents)
    cluster_rank_index = []

    for c in ranked_sents:
        indices_cluster = zip(*ranked_sents[c])[0]
        highl_positions = np.in1d (indices_cluster, indices_highl)
        
        ranking_pos = np.where (highl_positions)[0]

        cluster_rank_index += zip ([c]*ranking_pos.shape[0], 
            ranking_pos, np.array(indices_cluster)[highl_positions])

    return cluster_rank_index


def load_weights (file_name, alpha, n_attributes):
    
    weights = pd.read_csv (file_name, sep='\t', encoding='utf-8')

    weights = weights [weights.alpha == alpha].drop(columns=['alpha'])
    weights = np.asarray (weights)[0]

    n_cols_tfidf = n_attributes - weights.shape[0] + 1
    array_tfidf = np.full (n_cols_tfidf, weights[-1]/n_cols_tfidf)

    return np.concatenate ( (weights[0:-1], array_tfidf) ) 
 

if __name__ == "__main__":
        
    dir_articles   = os.path.join (os.path.dirname(__file__), 'files_txt')

    list_of_files_no_dir = os.listdir (dir_articles)
    list_of_files = [ os.path.join (dir_articles, f) for f in list_of_files_no_dir ]

    #for f_index in range(5):
    file_name = os.path.join (dir_articles, 'rosasromero2016.txt')
    file_weights = file_name.replace('.txt', '_weights_3.csv').replace(
                                        'files_txt', 'files_results')

    sentences, sents_vecs, _ = tokenizer.map_article_to_vec(
                                file_name, highlights=True, bag_of_words=True)

    print len (sentences)

    highl_file_name = file_name.replace('files_txt', 'files_highlights')
    n_highlights = len(tokenizer.get_highlights (highl_file_name))

    weights = load_weights (file_weights, 1.5/3.0, sents_vecs.shape[1])


    labels, centroids = cluster_sentences (sents_vecs.values, 
                                            n_highlights, weights=weights)
    ranked_sents = rank_sentences (sents_vecs.values, labels, centroids)
    
    file_save = file_name.replace('files_txt', 'files_tests')

    ranked_highlights = get_ranking_highlights (sentences, ranked_sents, 
                                                n_highlights)
    mark_highlights (sentences, n_highlights, file_name)

    try:
        save_ranking_file (ranked_sents, sentences, ranked_highlights, file_save)
        print 'Arquivo "' + file_save + '" salvo com sucesso.'
    except IOError:
        print 'Erro ao salvar arquivo "' + file_save + '".'