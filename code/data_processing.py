from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from gensim import corpora
import os


def load_documents(path_to_documents):
    files = os.listdir(path_to_documents)
    docs = []
    for f in files:
        with open(os.path.join(path_to_documents, f), 'r', encoding='utf-8') as f_reader:
            docs.append(f_reader.read())
    return docs


def docs2matrix(docs):
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
        split_words = " ".join([i for i in document.split() if i not in self.stopwords])
        remove_punct = "".join(i for i in split_words if i not in self.punctuation)
        lemmas = " ".join(self.lemmatize.lemmatize(i) for i in remove_punct.split())
        tokenized = lemmas.split(" ")
        return tokenized

    def clean_docs(self, docs):
        return [self.cleaning(doc) for doc in docs]






