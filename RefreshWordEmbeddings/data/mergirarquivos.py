import os
import random
import numpy as np

all_files = os.listdir('labels')
random.shuffle(all_files)


def score_sentence(sent_rouge):
    return float(sent_rouge.split(' - ')[0])


def save_multipleoracle(filename, summary_indexes, rouge_score, n_sents):
    str_indexes = " ".join([str(i) for i in summary_indexes])
    with open('./multipleoracle/%s' % filename, 'w') as f:
        f.write('%d\n' % n_sents)
        f.write('%s %s' % (str_indexes, rouge_score))
        f.close()


def generate_multipleoracle(all_files):
    for filename in all_files:
        # Ler valor do rouge
        rouge_file = open("./rouge/%s" % filename).read().split('\n\n')[1]
        sentences_rouge = rouge_file.split('\n')
        rouge_scores = np.array([score_sentence(sent_rouge) for sent_rouge in
                                sentences_rouge])
        labels = np.array(open("./labels/%s" % filename).read().split('\n'))
        summary_indexes = list(np.where(labels == '1')[0])
        rouge_mean = rouge_scores[summary_indexes].mean()
        save_multipleoracle(filename, summary_indexes, rouge_mean, len(labels))


def create_data(filelist, data_type, doc_or_label, dirname):
    with open('paperlist.%s.%s' % (data_type, doc_or_label), 'w') as f:
        for filename in filelist:
            content = open(os.path.join(dirname, filename)).read()
            f.write(filename + '\n')
            f.write(content.strip('\n') + '\n\n')
    f.close()


generate_multipleoracle(all_files)

create_data(all_files[:160], 'training', 'doc', './my_papers')
create_data(all_files[:160], 'training', 'label.singleoracle', './labels')
create_data(all_files[:160], 'training', 'label.multipleoracle',
            './multipleoracle')
create_data(all_files[160:180], 'validation', 'doc', './my_papers')
create_data(all_files[160:180], 'validation', 'label.singleoracle', './labels')
create_data(all_files[160:180], 'validation', 'label.multipleoracle',
            './multipleoracle')
create_data(all_files[180:], 'test', 'doc', './my_papers')
create_data(all_files[180:], 'test', 'label.singleoracle', './labels')
create_data(all_files[180:], 'test', 'label.multipleoracle', './multipleoracle'
            )
