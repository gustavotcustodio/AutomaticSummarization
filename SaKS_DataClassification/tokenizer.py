from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.kmeans import KMeansClusterer
from scipy.spatial import distance
from stemming.porter2 import stem
import pandas as pd
import numpy as np
import re
import os
import io
import math
import functools
import pdb; pdb.set_trace()  # XXX BREAKPOINT


def get_max_number_keywords(list_of_keywords):
    n_keywords = []

    for keywords in list_of_keywords:
        n_keywords.append(len(keywords.split(',')))

    return max(n_keywords)


def get_words_frequency(full_text):
    # Counting words

    words_no_symbols = clean_text_and_tokenize(full_text)

    final_words, count = np.unique(words_no_symbols, return_counts=True)

    count = map(lambda n: float(n)/len(final_words), count)

    return zip(final_words, count)


def clean_text_and_tokenize(text):
    words = word_tokenize(text)

    words_lower = map(lambda w: w.lower(), words)
    words_no_stop = filter(lambda w: w not in stopwords.words('english'),
                           words_lower)
    words_no_symbols = filter(re.compile(r'[a-z1-9].*').search, words_no_stop)

    return map(stem, words_no_symbols)


def sum_word_freq(words_in_sentence, word_freq):
    # Sum the frequency of words in a sentence
    n_words = len(words_in_sentence)

    sum_freq = sum([word_freq[w]/n_words for w in words_in_sentence
                    if w in word_freq])
    return sum_freq


def get_keywords_similarity(words_in_sentence, keywords):
    keywords_match = []

    for words in keywords:
        matches = map(lambda w: 1 if w in words_in_sentence else 0, words)
        keywords_match.append(2.0 * sum(matches) / (
            len(words) + len(words_in_sentence)))
    return keywords_match


def get_section_similarity(words_in_sentence, words_in_section):
    matches = map(lambda w: 1 if w in words_in_section else 0,
                  words_in_sentence)
    if len(matches) <= 0:
        return 0
    return 2.0 * sum(matches)/(len(words_in_sentence) + len(words_in_section))


def get_title(text):
    return text.split('\n')[0]


def get_highlights(file_path):
    """ Read the txt file with the research highlights of the respective files
    """
    text_file = io.open(file_path, mode='r', encoding='utf-8')
    highlights = text_file.read().split('\n')
    # highlights = '^~_'.join(text_file.read().split('\n'))
    text_file.close()
    return highlights


def get_session_lines(text, session):
    lines = text.split('\n')

    if session == 'a':  # abstract
        r_start = re.compile("^Abstract$")
        r_end = re.compile("Keywords|Abbreviations")

    elif session == 'i':  # introduction
        r_start = re.compile(r'1.\s+Introduction\s*')
        r_end = re.compile(r'2.\s+[A-Z0-9][a-zA-Z0-9]+.*')

    else:  # conclusion
        r_start = re.compile(r'[1-9][0-9]?.\s+(Conclu.*|Discussion.*|Summary'
                             '*|.*conclu.*|.*future.*.|Results.*|Final.*)')
        r_end = re.compile(r'(Append.*|^1$)')
    session_lines = []
    candidate_sessions = []
    found_session = False

    for i in range(len(lines)):
        if r_start.match(lines[i]):
            candidate_sessions.append(i)
            found_session = True
    if found_session:
        session_lines.append(candidate_sessions[-1])
        i = session_lines[0] + 1

        while i < len(lines) and not(r_end.match(lines[i])):
            session_lines.append(i)
            i += 1
    return session_lines


def extract_keywords(text):
    """ After finding the string "Keywords", each line
    is a keyword until an empty line is found """
    keywords = list()
    reading_keywords = False
    all_lines = text.split('\n')

    for line in all_lines:
        if 'Keywords' in line:
            reading_keywords = True

        # nothing in line
        elif not line and reading_keywords:
            return ','.join(keywords)

        elif reading_keywords:
            keywords.append(line)
    return ','.join(keywords)


def extract_content(path):
    """
    Extracts the keywords, highlights and the text in a article
    'path': name of the file
    """
    article = io.open(path, mode="r", encoding="utf-8")

    abstract, introduction, conclusion, final_text = '', '', '', ''

    full_text = article.read()
    full_text_split = np.array(full_text.split('\n'))

    abstract_lines = get_session_lines(full_text, 'a')
    abstract = '\n'.join(full_text_split[abstract_lines])

    # get the lines containing the introduction
    intro_lines = get_session_lines(full_text, 'i')
    introduction = '\n'.join(full_text_split[intro_lines])

    text_without_intro = '\n'.join(full_text_split[(intro_lines[-1]+1):])
    text_without_intro_split = np.array(text_without_intro.split('\n'))

    conclu_lines = get_session_lines(text_without_intro, 'c')

    if conclu_lines:
        conclusion = '\n'.join(text_without_intro_split[conclu_lines])

        text_without_conclu_1 = '\n'.join(text_without_intro_split[
            0:conclu_lines[0]])
        text_without_conclu_2 = '' if(conclu_lines[-1]+1) >= \
            len(text_without_intro_split) else \
            '\n'.join(text_without_intro_split[(conclu_lines[-1]+1):])

        final_text = text_without_conclu_1 + text_without_conclu_2
    else:
        final_text = text_without_intro

    return get_title(full_text), extract_keywords(full_text), abstract, \
        introduction, conclusion, final_text


def create_sentences_table(list_of_files, highlights=False):
    if highlights:
        cols = ['title', 'keywords', 'abstract', 'introduction', 'conclusion',
                'text', 'highlights']
        df = pd.DataFrame([list(extract_content(f)) + [get_highlights(f)]
                           for f in list_of_files], columns=cols)
    else:
        cols = ['title', 'keywords', 'abstract', 'introduction', 'conclusion',
                'text']
        df = pd.DataFrame([list(extract_content(f)
                                ) for f in list_of_files], columns=cols)

    df.to_csv("articles_highlights.csv", sep='\t', encoding='utf-8',
              index=False)


def calc_df(word, sentences):
    n_sents_with_word = 0
    for sent in sentences:
        n_sents_with_word += 1 if word in sent else 0
    return n_sents_with_word


def calc_tf_idf_word(word, sentences):
    df = calc_df(word, sentences)
    N = len(sentences)

    tfidf_vals = []

    for sent in sentences:
        tf = float(sent.count(word)) / len(sent)

        idf = math.log(float(N) / df)

        tfidf_vals.append(tf * idf)

    return np.array(tfidf_vals)


def create_bag_of_words(tokenized_sentences):

    word_list = np.concatenate(tokenized_sentences)
    word_list = np.unique(word_list)

    n_sents = len(tokenized_sentences)
    n_words = word_list.shape[0]

    bag_of_words = np.zeros((n_sents, n_words))
    for w in range(n_words):
        bag_of_words[:, w] = calc_tf_idf_word(word_list[w],
                                              tokenized_sentences)
    return bag_of_words


def create_sents_vector(tokenized_sentences, sentences_vectors,
                        sents_in_section, keywords, word_freq):
    for s in tokenized_sentences:
        # Add sentence to the cluster

        keywords_match = get_keywords_similarity(s, keywords)

        # get sentence's degree of similarity with the abstract
        abstract_match = get_section_similarity(s, functools.reduce(
            lambda x, y: x+y, sents_in_section['abstract']))

        intro_match = get_section_similarity(s, functools.reduce(
            lambda x, y: x+y, sents_in_section['introduction']))

        text_match = get_section_similarity(s, functools.reduce(
            lambda x, y: x+y, sents_in_section['text']))

        conclu_match = get_section_similarity(s, functools.reduce(
            lambda x, y: x+y, sents_in_section['conclusion']))

        # sum of freq. of words in the sentence
        word_freq_sentence = sum_word_freq(s, word_freq)

        index = len(sentences_vectors)
        sentences_vectors.loc[index] = [abstract_match] + keywords_match + \
            [intro_match, text_match, conclu_match, word_freq_sentence]


def cluster_sents(sents_vecs, n_clusters):
    kclusterer = KMeansClusterer(n_clusters, repeats=1,
                                 distance=distance.euclidean,
                                 avoid_empty_clusters=True)
    labels = kclusterer.cluster(sents_vecs.values, assign_clusters=True)

    centroids = np.array(kclusterer.means())

    return np.array(labels), centroids


def count_num_sents_cluster(sents_vectors, sections_sents, n_clusters):
    """
    Cluster sentences and count the number of times that sentences from each
    section appear in each cluster.
    Ex: 4 sents from introduction and 3 sentences from conclusion in cluster x.
    """
    labels, centroids = cluster_sents(sents_vectors, n_clusters)
    sections = ['abstract', 'introduction', 'conclusion', 'text']

    sents_cluster_values = []
    n_sents_by_cluster = []

    for c in range(n_clusters):
        n_sents = {}

        for sec in sections:
            n_sents[sec] = 0.0

        # Get indices in c cluster
        indices_cluster = np.where(labels == c)[0]

        for i in indices_cluster:
            if sections_sents[i] != 'highlights':
                n_sents[sections_sents[i]] += 1

        n_sents_by_cluster.append(n_sents)

    for lbl in labels:
        sents_cluster_values.append(n_sents_by_cluster[lbl].values())

    columns = ['n_sents_intro', 'n_sents_text', 'n_sents_abst',
               'n_sents_conclu']
    return np.array(sents_cluster_values), columns


def map_article_to_vec(article_path, highlights=False):

    sections_content = list(extract_content(article_path))

    if highlights:
        path_highl = article_path.replace('files_txt', 'files_highlights')
        highlights = get_highlights(path_highl)

    sections_names = ['title', 'keywords', 'abstract', 'introduction',
                      'conclusion', 'text']
    content = dict(zip(sections_names, sections_content))

    n_keywords = len(content['keywords'].split(','))

    sentences_vectors = pd.DataFrame(columns=['abstract'] + [
        'keyword'+str(i+1) for i in range(n_keywords)] + sections_names[3::]
        + ['word freq. in sentence'])

    word_freq = dict(get_words_frequency('.'.join(sections_content)))

    all_sentences = []
    tokenized_sentences = []
    sents_in_section = {}
    sections_sents = []

    for col in ['abstract', 'introduction', 'text', 'conclusion']:
        sents_in_section[col] = sent_tokenize(content[col])
        token_section = map(clean_text_and_tokenize, sents_in_section[col])

        indices_valid_sents = get_valid_sents_indices(token_section)

        # Sections in which the sentences belong
        sections_sents += len(indices_valid_sents) * [col]

        tokenized_sentences += [token_section[i] for i in indices_valid_sents]

        sents_in_section[col] = [sents_in_section[col][i]
                                 for i in indices_valid_sents]
        all_sentences += sents_in_section[col]

    if highlights:
        all_sentences += highlights
        tokenized_sentences += map(clean_text_and_tokenize, highlights)
        sections_sents += len(highlights) * ['highlights']

    keywords = map(clean_text_and_tokenize, content['keywords'].split(','))

    create_sents_vector(tokenized_sentences, sentences_vectors,
                        sents_in_section, keywords, word_freq)

    normalize_cols(sentences_vectors)

    return all_sentences, sentences_vectors, tokenized_sentences


def normalize_cols(sents_vecs):
    for col in sents_vecs.columns:
        max_val = sents_vecs[col].max()
        min_val = sents_vecs[col].min()

        if (max_val - min_val) > 0:
            sents_vecs[col] = (sents_vecs[col] - min_val) / (max_val - min_val)
    return sents_vecs


def get_valid_sents_indices(token_sents):
    indices = []

    for i in range(len(token_sents)):
        if len(token_sents[i]) > 2:
            indices.append(i)

        elif len(token_sents[i]) == 2:
            word_1_not_num = not(re.match(r'^[0-9]\.*[0-9]*$',
                                          token_sents[i][0]))
            word_2_not_num = not(re.match(r'^[0-9]\.*[0-9]*$',
                                          token_sents[i][1]))
            if word_1_not_num and word_2_not_num:
                indices.append(i)
    return indices


def calc_similarity_matrix(token_sents):
    n_sentences = len(token_sents)

    similarity_matrix = np.zeros((n_sentences, n_sentences))

    for i in range(n_sentences):
        for j in range(n_sentences):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = get_section_similarity(
                                            token_sents[i], token_sents[j])
    return similarity_matrix


def mark_highlights(sents_vecs, n_highlights):
    n_sents = sents_vecs.shape[0]

    highlight_indicator = (n_sents - n_highlights)*[0.0] + n_highlights*[1.0]
    sents_vecs.insert(0, 'is_a_highlight', highlight_indicator)


def save_similarity_matrix(similarity_matrix, file_name):
    np.savetxt(file_name, similarity_matrix, delimiter=';')


if __name__ == "__main__":
    dir_txts = os.path.join(os.path.dirname(__file__), 'files_txt')
    list_of_files_no_dir = os.listdir(dir_txts)
    list_of_files = [os.path.join(dir_txts, f) for f in list_of_files_no_dir]

    for f in list_of_files:

        sentences, sents_vecs, token_sents = map_article_to_vec(f, True)

        file_name = f.replace('.txt', '.csv').replace(
            'files_txt', 'files_cluster_values_2')

        highlights = get_highlights(f.replace('files_txt', 'files_highlights'))
        mark_highlights(sents_vecs, len(highlights))

        sents_vecs.to_csv(file_name, sep='\t', encoding='utf-8', index=False)

        print('Arquivo ' + file_name + ' salvo.')
