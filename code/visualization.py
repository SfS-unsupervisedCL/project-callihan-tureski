import numpy as np
from numpy import isnan
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import output_file, save
from bokeh.models import ColumnDataSource, HoverTool
import logging
import pyLDAvis.gensim as gensimvis
import pyLDAvis




class Visualize:

    def __init__(
            self,
            num_categories=20,
            threshold=0.0,
            n_top_words=5,
            n_iterations=500,
            language='english'
    ):
        self.num_categories = num_categories
        self.threshold = threshold
        self.n_top_words = n_top_words
        self.n_iterations = n_iterations
        self.language = language

    def visualize(self, ldamodel, doc_matrix, raw_documents):
        """
        Visalizes predictions from LDA model. Uses t-SNE to reduce dimensionality of model.
        Saves an HTML file with interactive graph which separates classes by color and provides a preview of the
        document when moused over.
        :param ldamodel:
        :param doc_matrix:
        :param raw_documents:
        :return:
        """
        
        prob_matrix = np.zeros((len(doc_matrix), self.num_categories))

        for i, doc in enumerate(doc_matrix):
            predictions = ldamodel[doc]
            idx, prob = zip(*predictions)
            prob_matrix[i, idx] = prob

        _idx = np.amax(prob_matrix, axis=1) > self.threshold  # idx of news that > threshold
        _topics = prob_matrix[_idx]

        num_example = len(_topics)

        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                          init='pca')
        tsne_lda = tsne_model.fit_transform(_topics[:num_example])

        # find the most probable topic for each news
        _lda_keys = []
        for i in range(_topics.shape[0]):
            _lda_keys += _topics[i].argmax(),

        # show topics and their top words
        topic_summaries = []
        for i in range(self.num_categories):
            word, _ = zip(*ldamodel.show_topic(i, topn=self.n_top_words))
            topic_summaries.append(' '.join(word))

        # 20 colors
        # colormap = np.array([
        #   "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        #   "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        #   "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        #   "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
        # ])

        colormap = np.array(['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2',
                             '#31a354', '#74c476', '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb',
                             '#636363', '#969696', '#bdbdbd', '#d9d9d9'])

        while self.num_categories > len(colormap):
            colormap = np.append(colormap, colormap)
            logging.info("CM Len:", len(colormap))

        title = "t-SNE visualization of LDA model trained on {} news, " \
                "{} topics, thresholding at {} topic probability, {} iter ({} data " \
                "points and top {} words)".format(prob_matrix.shape[0], self.num_categories, self.threshold,
                                                  self.n_iterations, num_example, self.n_top_words)

        p = bp.figure(plot_width=1400, plot_height=1100, title=title,
                      tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                      x_axis_type=None, y_axis_type=None, min_border=1)

        source = ColumnDataSource(data=dict(
            x=tsne_lda[:, 0],
            y=tsne_lda[:, 1],
            color=colormap[_lda_keys][:num_example],
            content=raw_documents[:num_example],
            topic_key=_lda_keys[:num_example]
            )
        )

        p.scatter(x='x', y='y', color='color', source=source)

        topic_coord = np.empty((prob_matrix.shape[1], 2)) * np.nan
        for topic_num in _lda_keys:
            if not np.isnan(topic_coord).any():
                break
            topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]


        where_are_NaNs = isnan(prob_matrix)
        prob_matrix[where_are_NaNs] = 0
        where_are_NaNs = isnan(topic_coord)
        topic_coord[where_are_NaNs] = 0
        # plot crucial words
        logging.info("Prob Mat", prob_matrix.shape)
        for i in range(prob_matrix.shape[1]):
            p.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

        # hover tools
        hover = p.select(dict(type=HoverTool))
        hover.tooltips = {"content": "@content - topic: @topic_key"}

        output_file("%s_%s_categories.html" % (self.language, str(self.num_categories)))
        save(p)

    def visualize_with_pydavis(self, ldamodel, corpus, train_term_dictionary):

        vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, train_term_dictionary)
        pyLDAvis.display(vis_data)




        return