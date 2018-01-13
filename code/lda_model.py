import time
from gensim.models.ldamodel import LdaModel
from time import time


class Model:

    def __init__(
            self,
            n_iterations=500,
            n_top_words=5,
            threshold=0.0,
    ):
        self.n_iterations = n_iterations
        self.n_top_words = n_top_words
        self.threshold = threshold
        self.n_categories = 20

    def create_model(self,
                     doc_matrix,
                     term_dictionary,
                     n_categories=20,
                     save_model=False,
                     language='lang'):

        start = time()
        if n_categories != 20:
            self.n_categories = n_categories

        ldamodel = LdaModel(doc_matrix,
                            num_topics=self.n_categories,
                            id2word=term_dictionary,
                            passes=50)

        if save_model:
            ldamodel.save('%s_%d_lda.model' % (language, self.n_categories))
            print("Model saved")

        print('Training lasted: {:.2f}s'.format(time()-start))

        return ldamodel

    def load_model(self, model_path='lda.model'):

        return LdaModel.load('topic.model')


