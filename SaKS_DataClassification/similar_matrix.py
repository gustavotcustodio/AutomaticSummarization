import re
import os
import numpy as np
from stemming.porter2 import stem
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import tokenizer


def clean_text_and_tokenize(text):
    """
    Remove stopwords, numbers and symbols, tokenize the sentence
    in words and get the word-stem of the remaining words.

    Parameters
    ----------
    text: string
        Sentence that we wish to clean and split in words.

    Returns
    -------
    word_stems: list[string]
        List of word stems after the cleaning process is finished.

    """
    words = word_tokenize(text)
    words_lower = map(lambda w: w.lower(), words)
    words_no_stop = filter(lambda w: w not in stopwords.words('english'),
                           words_lower)
    words_no_symbols = filter(re.compile(r'[a-z1-9].*').search, words_no_stop)

    return map(stem, words_no_symbols)


def get_sentences_similarity(words_in_sentence_1, words_in_sentence_2):
    """
    Calculate the similarity between two sentences by the number of words in
    common.

    Parameters
    ----------
    words_in_sentence_1: list [string]
        First sentence to compare.
    words_in_sentence_2: list [string]
        Second sentence to compare.

    Returns
    -------
    similarity: float
        Value between 0 and 1 that gives the similarity between two sentences.

    """
    matches = map(lambda w: 1 if w in words_in_sentence_1 else 0,
                  words_in_sentence_2)

    if len(matches) <= 0:
        return 0

    return 2.0 * sum(matches) / (len(words_in_sentence_1) +
                                 len(words_in_sentence_2))


def calc_similarity_matrix(token_sents):
    n_sentences = len(token_sents)

    similarity_matrix = np.zeros((n_sentences, n_sentences))

    for i in range(n_sentences):
        for j in range(n_sentences):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = get_sentences_similarity(
                                            token_sents[i], token_sents[j])
    return similarity_matrix


def save_similarity_matrix(similarity_matrix, file_name):
    np.savetxt(file_name, similarity_matrix, delimiter=';')


if __name__ == "__main__":
    dir_txts = os.path.join(os.path.dirname(__file__), 'files_txt')
    list_of_files = [os.path.join(dir_txts, f) for f in os.listdir(dir_txts)]

    for f in list_of_files:
        print(f)

        _, _, token_sents = tokenizer.map_article_to_vec(f)

        similarity_matrix = calc_similarity_matrix(token_sents)

        file_name = f.replace('.txt', '.csv'
                              ).replace('files_txt', 'files_sim_matrices')

        save_similarity_matrix(similarity_matrix, file_name)
