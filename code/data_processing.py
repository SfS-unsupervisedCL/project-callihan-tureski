from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
import os
from chinese_processing import ChineseStopwords

def load_texts_from_directory(path_to_documents, subset=None):
    """
    Loads all .txt files from directory into a list.
    :param path_to_documents:
    :return list of documents:
    """
    files = os.listdir(path_to_documents)
    if subset is not None:
        files = files[subset[0]:subset[1]]
    docs = []
    keywords = dict()
    for f in files:
        doc = ''
        with open(os.path.join(path_to_documents, f), 'r', encoding='utf-8') as file:
            for i, l in enumerate(file.readlines()):
                if i is 0:
                    keywords[f.split('.')[0]] = l.replace('%%%','').strip().split('|')[:-1]
                else:
                    doc += l.strip() + ' '
            file.close()
        docs.append(doc)
    return docs, keywords


def docs2matrix(docs):
    """
    Transforms a list of documents into a bag of words matrix suitable for the LDA model.
    :param docs:
    :return bag of words matrix:
    """
    # [token for doc in docs for token in doc]
    term_dictionary = corpora.Dictionary(docs)
    doc_matrix = [term_dictionary.doc2bow(doc) for doc in docs]
    print("Len of raw corpus: %i | Len of matrix: %i" % (len(docs), len(doc_matrix)))
    return doc_matrix, term_dictionary

class Processing:
    def __init__(
            self,
            stopword_lang='english'
    ):
        self.lang = stopword_lang
        if stopword_lang is 'chinese':
            self.stopwords = set(ChineseStopwords().chinese_stopwords)
        else:
            self.stopwords = set(stopwords.words(stopword_lang))
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()

    def cleaning(self, document):
        """
        Cleans document of stopwords and punctuation. Stopwords are specified at initialization of Processing.
        Lemmatizes for all languages except Chinese.
        :param document:
        :return tokenized and cleaned document:
        """
        split_words = [i for i in list(document) if i not in self.stopwords]
        remove_punct = "".join(i for i in split_words if i not in self.punctuation)
        if self.lang is not 'chinese':
            # Lemmatizes if not chinese
            remove_punct = " ".join(self.lemmatize.lemmatize(i) for i in remove_punct.split())
        tokenized = remove_punct.split(" ")
        return tokenized

    def clean_docs(self, docs):
        """
        Cleans all documents in a list
        :param docs:
        :return list of cleaned documents:
        """
        return [self.cleaning(doc) for doc in docs]






