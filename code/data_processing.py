from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
import os
import numpy as np
np.set_printoptions(threshold=np.nan)
from chinese_processing import ChineseStopwords
import logging
from sklearn.cluster import KMeans


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

    def cluster_data(self, doc_matrix, ldamodel, to_csv=False, keywords=None, filenames=None, num_categories=-1):
        """
        Gets cluster data. Writes to CSV if to_csv=True
        :param doc_matrix:
        :param ldamodel:
        :param to_csv:
        :param keywords:
        :param filenames:
        :param num_categories:
        :return:
        """
        test_clusters = []
        for doc in doc_matrix:
            scores = ldamodel[doc]
            # TODO check argmax
            test_clusters.append(scores[np.argmax([s[1] for s in scores])][0])
        if to_csv and keywords is not None and filenames is not None and num_categories is not -1:
            filename = '%s_%d_categories.csv' % (self.lang, num_categories)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('file,language,num_categories,cluster,keywords')
                for i, fn in enumerate(filenames):
                    f.write('\n%s,%s,%d,%d,%s' % (
                        fn,
                        self.lang,
                        num_categories,
                        test_clusters[i],
                        '|'.join(keywords[i])
                    ))
        return test_clusters

    def kmeans_cluster_data(self, doc_matrix, ldamodel, num_categories):
        """
       Clusters documents based on their probability distributions for topics as returned by the LDA model.
       :param doc_matrix:
       :param ldamodel:

       :return data as clustered thing:
       """

        topic_probs_per_doc = np.zeros(shape=(len(doc_matrix), num_categories), dtype=np.float64)
        for index, doc in enumerate(doc_matrix):
            scores = ldamodel[doc]

            for count, score in enumerate(scores):
                topic_probs_per_doc[index, score[0]] = score[1]


        # print(test_clusters[:10])
        kmeans = KMeans(n_clusters=num_categories, random_state=0).fit(topic_probs_per_doc)

        print("K means complete")

        return kmeans

