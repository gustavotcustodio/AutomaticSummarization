from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import file_reader


def run_one_class_svm(X_normal, X_highl, file_name):

    n_folds = X_highl.shape[0]

    print('\n=========================== ' + file_name +
          ' ===========================')

    clf = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)

    grid_param = {'nu': [0.1, 0.2, 0.5, 0.8],
                  'kernel': ['rbf', 'linear'],
                  'gamma': [0.1, 0.25, 0.5]}

    y = X_highl.shape[0] * [1]
    gd_sr = do_grid_search(X_highl, y, clf, grid_param, n_folds)

    y_pred = gd_sr.predict(X_normal)

    print('Number of non-highlights classified as highlights: ')
    nonhighlights = len(np.where(y_pred == 1)[0])
    allsentences = X_normal.shape[0]
    print('%d of %d sentences' % (nonhighlights, allsentences))
    print(nonhighlights / float(allsentences))


def do_grid_search(X, y, clf, grid_param, n_folds):
    gd_sr = GridSearchCV(estimator=clf, param_grid=grid_param,
                         scoring='accuracy', cv=n_folds)
    gd_sr.fit(X, y)
    n_sets_params = len(gd_sr.cv_results_['std_train_score'])

    best_results_train = np.zeros((n_folds, n_sets_params))
    best_results_test = np.zeros((n_folds, n_sets_params))

    for i in range(n_folds):
        best_results_train[i] = gd_sr.cv_results_[
                            'split' + str(i) + '_train_score']

        best_results_test[i] = gd_sr.cv_results_[
                            'split' + str(i) + '_test_score']

    for i in range(n_sets_params):

        print(gd_sr.cv_results_['params'][i])
        print('Training')
        print(best_results_train[:, i])
        print('Test')
        print(best_results_test[:, i])
        print

    print('best params: ')
    print(gd_sr.best_params_)
    print
    return gd_sr


def run_training_test(data, labels, n_folds, algorithm):
    Xy = np.column_stack((data, labels))

    np.random.shuffle(Xy)

    X, y = Xy[:, 0:-1], Xy[:, -1]

    if algorithm == 'logistic_regression':
        clf = LogisticRegression(C=1, penalty='l2')
        grid_param = {'C': [1, 10, 100, 1000], 'tol': [1e-5, 1e-4, 1e-3]}

    elif algorithm == 'svm':
        clf = SVC(gamma=1e-3, kernel='rbf')
        grid_param = {'C': [1, 10, 100, 1000],
                      'gamma': [1e-3, 1e-4], 'kernel': ['rbf', 'linear']}

    elif algorithm == 'random_forest':
        clf = RandomForestClassifier(n_estimators=50, criterion='gini')
        grid_param = {'n_estimators': [50, 100, 200, 300],
                      'criterion': ['gini', 'entropy']}

    elif algorithm == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
        grid_param = {'n_neighbors': [3, 5, 7], 'p': [1, 2]}

    elif algorithm == 'naive_bayes':
        clf = GaussianNB()
        grid_param = {}

    do_grid_search(X, y, clf, grid_param, n_folds)


def run_classification(data, labels):
    ind_highl = np.where(labels == 1)[0]
    np.random.shuffle(ind_highl)

    ind_normal = np.where(labels == 0)[0]
    np.random.shuffle(ind_normal)
    ind_normal = ind_normal[0:ind_highl.shape[0]]

    print(ind_normal)

    data = np.vstack((data[ind_normal], data[ind_highl]))
    labels = np.hstack((labels[ind_normal], labels[ind_highl]))

    n_folds = 10

    print('------------------- Naive Bayes -------------------')
    run_training_test(data, labels, n_folds, 'naive_bayes')

    print('------------------- Logistic Regression -------------------')
    run_training_test(data, labels, n_folds, 'logistic_regression')

    print('------------------- Random Forest -------------------')
    run_training_test(data, labels, n_folds, 'random_forest')

    print('------------------- KNN -------------------')
    run_training_test(data, labels, n_folds, 'knn')

    print('------------------- SVM -------------------')
    run_training_test(data, labels, n_folds, 'svm')


if __name__ == '__main__':
    path_files = os.path.join(os.path.dirname(__file__), 'files_cluster_values'
                              )
    files_names = os.listdir(path_files)

    files_names = map(lambda f: os.path.join(path_files, f), files_names)
    job = 'one_class'

    if job == 'classification':
        data = np.array([])
        labels = np.array([]).reshape((0,))

        for f in files_names:
            X, y = file_reader.load_train_and_test(f)

            data = file_reader.add_to_data(X, data)
            labels = np.hstack((labels, y))

        run_classification(data, labels)
    else:
        for f in files_names[5:10]:
            X, y = file_reader.load_train_and_test(f)

            X_normal = X[np.where(y == 0)[0]]
            X_highl = X[np.where(y == 1)[0]]

            run_one_class_svm(X_normal, X_highl, os.path.split(f)[-1])
