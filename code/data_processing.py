from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
import os
import numpy as np
from chinese_processing import ChineseStopwords
import logging


def language_vectors(categories, argmax=False):
    """
    Takes in a list of category distributions and returns vectors which correspond to the number of
    categories which are number of documents long. (num_categories, num_documents)
    If you do not want a category distribution, but rather a single category per document, set argmax to true.
    :param categories:
    :param argmax:
    :return:
    """
    if argmax:
        idx = [np.argmax(x) for x in categories]
        categories = np.zeros((len(categories), len(categories[0])))
        categories[np.arange(len(categories)), idx] = 1
    return list(zip(*categories))


def cluster_data(doc_matrix, ldamodel, num_categories):
    """
    Gets cluster data. returns a list of probability distributions for clusters
    :param doc_matrix:
    :param ldamodel:
    :param num_categories:
    :return:
    """
    clusters = []
    for doc in doc_matrix:
        idx, scores = zip(*ldamodel[doc])
        vec = np.zeros(num_categories)
        vec[list(idx)] = list(scores)
        clusters.append(vec)
    return clusters


def write_cluster_data(lang, num_categories, filenames, clusters, keywords):
    filename = '%s_%d_categories.csv' % (lang, num_categories)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('file,language,num_categories,cluster,keywords')
        for i, fn in enumerate(filenames):
            f.write('\n%s,%s,%d,%d,%s' % (
                fn,
                lang,
                num_categories,
                clusters[i],
                '|'.join(keywords[i])
            ))


def load_texts_from_directory(path_to_documents, subset=None):
    """
    Loads all .txt files from directory into a list.
    :param subset:
    :param path_to_documents:
    :return list of documents:
    """
    files = sorted(os.listdir(path_to_documents))
    if subset is not None:
        files = files[subset[0]:subset[1]]
    docs = []
    keywords = []
    filenames = []
    for f in files:
        filenames.append(f)
        doc = ''
        with open(os.path.join(path_to_documents, f), 'r', encoding='utf-8') as file:
            for i, l in enumerate(file.readlines()):
                if i is 0:
                    keywords.append(l.replace('%%%', '').strip().split('|')[:-1])
                else:
                    doc += l.strip() + ' '
            file.close()
        docs.append(doc)
    return docs, keywords, filenames


def docs2matrix(docs):
    """
    Transforms a list of documents into a bag of words matrix suitable for the LDA model.
    :param docs:
    :return bag of words matrix:
    """
    # [token for doc in docs for token in doc]
    term_dictionary = corpora.Dictionary(docs)
    doc_matrix = [term_dictionary.doc2bow(doc) for doc in docs]
    logging.info("Len of raw corpus: %d | Len of matrix: %d" % (len(docs), len(doc_matrix)))
    return doc_matrix, term_dictionary


class Processing:
    def __init__(
            self,
            stopword_lang='english'
    ):
        self.lang = stopword_lang
        if self.lang == 'chinese':
            self.stopwords = set(ChineseStopwords().chinese_stopwords)
        else:
            self.stopwords = set(stopwords.words(self.lang))
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()

    def cleaning(self, document):
        """
        Cleans document of stopwords and punctuation. Stopwords are specified at initialization of Processing.
        Lemmatizes for all languages except Chinese.
        :param document:
        :return tokenized and cleaned document:
        """
        remove_punct = ''.join(i for i in document.lower() if i not in self.punctuation)
        tokenized = [i for i in remove_punct.split() if i not in self.stopwords]
        if self.lang is not 'chinese':
            # Lemmatizes if not chinese
            tokenized = [self.lemmatize.lemmatize(i) for i in tokenized]
        return tokenized

    def clean_docs(self, docs):
        """
        Cleans all documents in a list
        :param docs:
        :return list of cleaned documents:
        """
        cleaned = [self.cleaning(doc) for doc in docs]
        print(cleaned[0])
        return cleaned
