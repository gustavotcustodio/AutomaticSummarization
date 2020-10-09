import matplotlib.pyplot as plt
import pandas as pd
import numpy as np   
import os


def create_table_weights (files_names, directory):
    '''
    
    '''
    weights = pd.DataFrame()
    file_number = 1

    for f in files_names:
        df = pd.read_csv (os.path.join (
                            directory , f), delimiter = "\t", encoding = 'utf-8')
        alphas = df.alpha.unique()
        for a in alphas:
            weights = weights.append (df.loc[df['alpha']==a].tail(1), ignore_index=True)
        file_number += 1

    return weights


def create_table_results (files_names, directory): 
    '''
    Read 5 experiment files for a article and returns
    a Dataframe with the data.

    Parameters
    ----------
    files_names: list[string]
        Names of the 5 experiment files for a same article. 
        Example: article_1.csv, article_2.csv, ..., article_5.csv.

    Returns
    -------
    results: DataFrame
        DataFrame containing the experiments results for the
        current article.
    '''

    results = pd.DataFrame()
    file_number = 1

    for f in files_names:
        df = pd.read_csv (os.path.join (
                            directory , f), delimiter = "\t", encoding = 'utf-8')
        df['run'] = [file_number] * df.shape[0]
        results = results.append (df, ignore_index=True)
        file_number += 1

    return results


def plot_disp_chart(data, alphas, runs, file_name, n_iters = 50, type='avg'):

    lines_labels = ['Silhouette', 'Avg. Distance Highlight-Centroid', 
                    'Objective Function']

    column = ['silhouettes', 'distances', 'evals']
    n_columns = len(column)

    x_label = 'Iter.'
    colors = ['green', 'blue', 'red']    

    if type=='avg':
        fig = plt.figure(figsize=(18, 7 ))
        fig.subplots_adjust(bottom=0.4)
    else:
        fig = plt.figure(figsize=(18, 25))

    col = 0
    for a in alphas:
        relevant_data = data.loc[np.isclose(data['alpha'], a)]
        title = 'Silhouette: ' + str(round(
                    a,2)) + ' . Avg. Distance: ' + str(1-round(a,2))

        if type=='avg':
            filtered_data = relevant_data.groupby(
                            'run').tail(n_iters).drop(columns=['run','alpha'])

            iterations = [i%n_iters for i in range(runs*n_iters)]
            filtered_data['iter'] = iterations

            mean = filtered_data.groupby('iter').mean()
            std  = filtered_data.groupby('iter').std ()
            n_clusters = filtered_data['n_clusters'].mean()

            plt.subplot (1, n_columns, col+1)

            for j in range(n_columns):
                plt.plot     (range(n_iters), mean[column[j]], 
                                color=colors[j], label=lines_labels[j])

                plt.errorbar (range(n_iters), mean[column[j]], 
                                yerr=std[column[j]], fmt='o', color=colors[j])

            plt.title ( title )
            plt.xlabel ( x_label )
            plt.text (0, 0.6, str(n_clusters)+' clusters',
                        verticalalignment='top', horizontalalignment='left')
            if col==1:
                plt.legend(lines_labels, bbox_to_anchor=(0.5, -0.2), loc="upper center")
        else:
            for row in range(0,runs):
                data_plot = relevant_data.loc [relevant_data['run']==(row+1)]
                total_iters = data_plot.shape[0]
                plt.subplot (runs, n_columns, row*n_columns+col+1)

                for j in range(n_columns):
                    plt.plot (range(total_iters), data_plot[column[j]], 
                                color=colors[j], label=lines_labels[j])
                
                plt.title ( title )
                plt.xlabel ( x_label )

                for pos_x in range(n_iters, total_iters, (n_iters+1)):
                    n_clusters = data_plot['n_clusters'].iloc[pos_x]

                    plt.text ( pos_x-n_iters, 0.6, str(n_clusters)+' clusters',
                                verticalalignment='top', horizontalalignment='left')

                    plt.axvline (x = pos_x, color='black', linestyle='dashed')

                if col==1 and row==4:
                    plt.legend(lines_labels, bbox_to_anchor=(0.5, -0.2), loc="upper center")

        col = (col+1) % 3         
        
    fig.savefig (file_name)  # save the figure to file
    plt.close(fig)


def draw_bar_chart (bars, n_clusters, alpha, row, col, 
                    n_charts_row=3, n_charts_col=5, std_dev=None):
    n_bars = bars.shape[0]
    x_labels = ['w'+str(i) for i in range(n_bars)]
    y_label  = 'Weight values (n. clusters = ' + str(n_clusters) + ')'

    title = 'Silhouette: ' + str(round(
                alpha,2)) + ' . Avg. Distance: ' + str(1-round(alpha,2))

    if std_dev is not None:
        plt.subplot  (1, n_charts_row, col)
    else:
        plt.subplot  (n_charts_col, n_charts_row, row*n_charts_row+col)

    plt.bar (x_labels, bars, 0.8, color='green')
    if std_dev is not None:
        plt.errorbar (x_labels, bars, yerr=std_dev, fmt='o')

    plt.title  (title)
    plt.ylabel (y_label)


def plot_bar_chart (data, alphas, file_name, type='avg'):
    
    if type=='avg':
        fig = plt.figure(figsize=(18, 4.5))
    else:
        fig = plt.figure(figsize=(18, 25 ))

    col = 0
    for a in alphas:
        relevant_data = data.loc[data['alpha']==a]
        filtered_data = relevant_data.drop (columns=['n_clusters','alpha'])

        if type=='avg':
            bars = filtered_data.mean()
            std  = filtered_data.std ()
            n_clusters = relevant_data['n_clusters'].mean()
            draw_bar_chart (bars, n_clusters, a, 0, col+1, std_dev=std)
        else:
            n_rows = 5
            bars = filtered_data
            for row in range (0, n_rows):
                n_clusters = relevant_data['n_clusters'].iloc[row]
                draw_bar_chart (bars.iloc[row], n_clusters, a, row, col+1)

        col = (col+1) % 3 

    fig.savefig (file_name)  # save the figure to file
    plt.close(fig)   
        

if __name__ == "__main__":

    directory = os.path.join (os.path.dirname(__file__), 'files_results')

    files_general = sorted (os.listdir (directory))

    files_evals   = filter (lambda e: '_evals_'   in e , files_general)
    files_weights = filter (lambda e: '_weights_' in e , files_general)

    while files_evals and files_weights:
        table_results = create_table_results (files_evals  [0:5], directory)
        table_weights = create_table_weights (files_weights[0:5], directory)

        cluster_values = table_results.drop_duplicates (
                            subset=['alpha','run'], keep='last')['n_clusters']

        table_weights['n_clusters'] = cluster_values.tolist()

        name_e = files_evals[0].replace('.csv','').strip('_1')
        name_w = files_weights[0].replace('.csv','').strip('_1')

        dir_charts = os.path.join (os.path.dirname(__file__), 'files_charts')

        file_name_eval   = os.path.join (os.path.join (
                            dir_charts, 'evals'),   name_e)
        file_name_weight = os.path.join (os.path.join (
                            dir_charts, 'weights'), name_w)


        #plot_bar_chart (table_weights, [1.0/ 2, 2.0/ 3, 1.0/ 3], 
        #                   file_name_weight + '_avg.pdf', type='avg')

        #plot_bar_chart (table_weights, [1.0/ 2, 2.0/ 3, 1.0/ 3], 
        #                   file_name_weight + '_multiple.pdf', type='multiple')

        plot_disp_chart (table_results, [1.0/ 2, 2.0/ 3, 1.0/ 3],
                            5, file_name_eval + '_avg.pdf', type='avg')

        plot_disp_chart (table_results, [1.0/ 2, 2.0/ 3, 1.0/ 3],
                            5, file_name_eval + '_multiple.pdf', type='multiple')

        files_evals   = files_evals   [5::]
        files_weights = files_weights [5::]