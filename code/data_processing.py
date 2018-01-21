from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
import os


def load_texts_from_directory(path_to_documents):
    """
    Loads all .txt files from directory into a list.
    :param path_to_documents:
    :return list of documents:
    """
    files = os.listdir(path_to_documents)
    docs = []
    for f in files:
        with open(os.path.join(path_to_documents, f), 'r', encoding='utf-8') as f_reader:
            docs.append(f_reader.read())
    return docs


def docs2matrix(docs):
    """
    Transforms a list of documents into a bag of words matrix suitable for the LDA model.
    :param docs:
    :return bag of words matrix:
    """
    print(docs[1])
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
        self.stopwords = set(stopwords.words(stopword_lang))
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()

    def cleaning(self, document):
        """
        Cleans document of stopwords and punctuation. Stopwords are specified at initialization of Processing.
        :param document:
        :return tokenized and cleaned document:
        """
        split_words = " ".join([i for i in document.split() if i not in self.stopwords])
        remove_punct = "".join(i for i in split_words if i not in self.punctuation)
        lemmas = " ".join(self.lemmatize.lemmatize(i) for i in remove_punct.split())
        tokenized = lemmas.split(" ")
        return tokenized

    def clean_docs(self, docs):
        """
        Cleans all documents in a list
        :param docs:
        :return list of cleaned documents:
        """
        return [self.cleaning(doc) for doc in docs]






