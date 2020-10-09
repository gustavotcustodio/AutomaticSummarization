import re
import os
import numpy as np
import pandas as pd
from lxml import etree
from stemming.porter2 import stem
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from config import xmls_dir, preprocessed_dir

ABSTRACT = 0
INTRODUCTION = 1
CONCLUSION = 2
OTHERS = 3


def get_keywords(xml_paper):
    return [keyword.text for keyword in
            xml_paper.find('keywords').findall('keyword')]


def get_section_sentences(xml_paper, section=OTHERS):
    if section == ABSTRACT:
        xml_abstract = xml_paper.find('abstract')
        return [sent.text for sent in xml_abstract.findall('sentence')]
    elif section == INTRODUCTION:
        return [sent.text for sent in xml_paper.findall('section')[0]]
    elif section == CONCLUSION:
        return [sent.text for sent in xml_paper.findall('section')[-1]]
    else:
        return [sent.text for xml_section in xml_paper.findall('section')[1:-1]
                for sent in xml_section.findall('sentence')]
    return []


def get_section_similarity(tokens_in_sentence, tokens_in_section):
    """ Computes the similarity of a paper' section with a sentence.

    Parameters
    ----------
    tokens_in_sentence: list
        All tokens in the sentence.
    tokens_in_section: list
        All tokens in the selected section.

    Returns
    -------
    float
        The similarity measure
    """
    matches = [1 if t in tokens_in_section else 0 for t in tokens_in_sentence]
    if len(matches) <= 0:
        return 0
    return 2.0*sum(matches) / (len(tokens_in_sentence)+len(tokens_in_section))


def calc_column_values(list_tokenized_sents, tokenized_sents_section):
    """ Get the attribute values of a single column in the attribute-value
    table.

    Parameters
    ----------
    list_tokenized_sents: list(list(string))
        List of tokens from each sentence in the paper (including highlights).
    tokenized_sents_section: list(list(string))
        List of tokens from each sentence in a specific section from the paper.

    Returns
    -------
    list(float)
        Values of a column in the attribute-value table.
    """
    tokens_in_section = [token for tokens_sentence in tokenized_sents_section
                         for token in tokens_sentence]

    return [get_section_similarity(tokens_in_sentence, tokens_in_section)
            for tokens_in_sentence in list_tokenized_sents]


def get_words_frequency(list_tokenized_sents):
    """ Get the frequency of each word inside the paper excluding keywords.

    Returns
    -------
    list(float)
        The sum of word frequency for each sentence.
    """
    all_words = np.array([token for tokenized_sentence in list_tokenized_sents
                          for token in tokenized_sentence])
    # Sum the number of occurences of a word in the text and divides by the
    # total number of words
    unique_words, count = np.unique(all_words, return_counts=True)
    freqs = [float(n) / len(unique_words) for n in count]
    dict_words_freqs = dict(zip(unique_words, freqs))

    # Use the dictionary to get the sum of word frequencies for each sentence.
    return [sum(dict_words_freqs[w] for w in tokenized_sentence)
            for tokenized_sentence in list_tokenized_sents]


def normalize(columns_values):
    """ Normalize the column values from 0 to 1. """
    max_val = max(columns_values)
    min_val = min(columns_values)
    if max_val == min_val:  # Avoid dividing by zero
        return columns_values
    return [(val-min_val) / (max_val-min_val) for val in columns_values]


def get_attribute_value_table(tokenized_keywords, tokenized_sents_abstract,
                              tokenized_sents_intro, tokenized_sents_conclu,
                              tokenized_sents_other):
    """ Creates an attribute-value table containing the similarity degree
    between a sentence and a section belonging to the paper, where each row
    represents a sentence and each column represents a section.
    """
    list_tokenized_sents = tokenized_sents_abstract + tokenized_sents_intro + \
        tokenized_sents_other + tokenized_sents_conclu
    attrib_value_table = {}
    attrib_value_table['abstract'] = calc_column_values(
        list_tokenized_sents, tokenized_sents_abstract)

    for i in range(len(tokenized_keywords)):
        attrib_value_table['keyword%d' % i] = calc_column_values(
            list_tokenized_sents, tokenized_keywords[i])

    attrib_value_table['introduction'] = calc_column_values(
        list_tokenized_sents, tokenized_sents_intro)
    attrib_value_table['conclusion'] = calc_column_values(
        list_tokenized_sents, tokenized_sents_conclu)
    attrib_value_table['text'] = calc_column_values(
        list_tokenized_sents, tokenized_sents_other)
    # Get the sum of word frequency in each sentence
    attrib_value_table['word freq. in sentence'] = get_words_frequency(
        list_tokenized_sents)
    # Normalize columns
    for col in attrib_value_table.keys():
        attrib_value_table[col] = normalize(attrib_value_table[col])
    return attrib_value_table


def clean_text_and_tokenize(text):
    words = word_tokenize(text)
    words_lower = map(lambda w: w.lower(), words)
    words_no_stop = filter(lambda w: w not in stopwords.words('english'),
                           words_lower)
    words_no_symbols = filter(re.compile(r'[a-z1-9].*').search, words_no_stop)
    return list(map(stem, words_no_symbols))


def preprocess():
    print('Starting preprocessing...')
    for xml in os.listdir(xmls_dir):
        with open(os.path.join(xmls_dir, xml)) as xmlreader:
            xml_paper = etree.parse(xmlreader)
            keywords = get_keywords(xml_paper)
            sentences_abstract = get_section_sentences(xml_paper, ABSTRACT)
            sentences_intro = get_section_sentences(xml_paper, INTRODUCTION)
            sentences_conclusion = get_section_sentences(xml_paper, CONCLUSION)
            other_sentences = get_section_sentences(xml_paper)

        # Get tokenized sentences for each section
        tokens_keywords = [clean_text_and_tokenize(k) for k in keywords]
        tokens_abstract = [clean_text_and_tokenize(s)
                           for s in sentences_abstract]
        tokens_intro = [clean_text_and_tokenize(s) for s in sentences_intro]
        tokens_conclu = [clean_text_and_tokenize(s)
                         for s in sentences_conclusion]
        tokens_other = [clean_text_and_tokenize(s) for s in other_sentences]

        # Get attribute-value table
        attrib_value_table = get_attribute_value_table(
            tokens_keywords, tokens_abstract, tokens_intro, tokens_conclu,
            tokens_other)
        # Saves the preprocessed text to a csv value
        outfile = os.path.join(preprocessed_dir, xml.replace('.xml', '.csv'))
        pd.DataFrame(attrib_value_table).to_csv(outfile)
        print('%s saved successfully.' % xml.replace('.xml', '.csv'))


if __name__ == '__main__':
    preprocess()
