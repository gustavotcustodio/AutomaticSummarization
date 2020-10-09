import numpy as np
import pandas as pd


def get_single_label(data, labels, lbl_value):
    indices = np.where(labels == lbl_value)[0]
    return data[indices]


def load_train_and_test(file_name):
    data = pd.read_csv(file_name, delimiter="\t", encoding='utf-8')
    X = data.values[:, 1:]
    y = data['is_a_highlight'].values
    return X, y


def add_to_data(X, data):
    if data.size == 0:
        return X

    # Number of attributes after keywords
    n_atribs_k = 8
    if data.shape[1] > X.shape[1]:
        n_extra_cols = data.shape[1] - X.shape[1]
        ind_complete = X.shape[1] - n_atribs_k

        zeros = np.zeros((X.shape[0], n_extra_cols))

        X_half = np.hstack((zeros, X[:, -n_atribs_k:]))
        X_full = np.hstack((X[:, :ind_complete], X_half))
        return np.vstack((data, X_full))
    elif data.shape[1] < X.shape[1]:
        n_extra_cols = X.shape[1] - data.shape[1]
        ind_complete = data.shape[1] - n_atribs_k

        zeros = np.zeros((data.shape[0], n_extra_cols))

        data_half = np.hstack((zeros, data[:, -n_atribs_k:]))
        data_full = np.hstack((data[:, :ind_complete], data_half))

        return np.vstack((data_full, X))

    else:
        return np.vstack((data, X))


def create_dataset(list_of_files, instances='all'):
    data = np.array([])
    labels = np.array([]).reshape((0,))
    for f in list_of_files:
        X, y = load_train_and_test(f)

        data = add_to_data(X, data)
        labels = np.hstack((labels, y))

    if instances == 'highlights':
        data = get_single_label(data, labels, 1)
        labels = np.array([1]*data.shape[0])

    elif instances == 'normal':
        data = get_single_label(data, labels, 0)
        labels = np.array([1]*data.shape[0])

    return data, labels
