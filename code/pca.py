import sys

sys.path.append("C:/Users/ryanc/Dropbox/school/usl_w18/project-callihan-mekki-tureski/code")

import os
from lda_model import Model
from data_processing import Processing, load_texts_from_directory, docs2matrix, cluster_data, language_vectors

"""
WRITTEN BY RYAN CALLIHAN
"""

ROOT = 'C:/Users/ryanc/Dropbox/school/usl_w18/project-callihan-mekki-tureski/code'
N_PASSES = 5
NUM_CATEGORIES = [5, 10, 20, 50]
LANGUAGES = ['english', 'arabic', 'russian', 'chinese']

vecs = dict()
for lang in LANGUAGES:

    vecs[lang] = dict()

    # Data processing class
    p = Processing(stopword_lang=lang)

    # Loads .txt files from specified directory
    documents, keywords, filenames = load_texts_from_directory(os.path.join(ROOT, 'data', lang))

    doc_matrix, term_dictionary = docs2matrix(p.clean_docs(documents))
    for num_cat in NUM_CATEGORIES:
        lda = Model(num_categories=num_cat)
        file = ''.join([lang, '_', str(num_cat), '_category_lda.model'])
        ldamodel = lda.load_model(model_path=os.path.join(ROOT, 'models', lang, file))

        clusters = cluster_data(doc_matrix, ldamodel, num_cat)

        vecs[lang][num_cat] = language_vectors(clusters)


import matplotlib.pyplot as plt
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D
import itertools

vecs_2d = dict()

colors = itertools.cycle(["r", "b", "g", "y"])



# for num_cat in NUM_CATEGORIES:

num_cat = 5
fig = plt.figure(1, figsize=(4, 3))
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

vecs_2d[num_cat] = dict()
for lang in LANGUAGES:

    pca = decomposition.PCA(n_components=2)
    pca.fit(vecs[lang][num_cat])
    vecs_2d[num_cat][lang] = pca.transform(vecs[lang][num_cat])
    plt.scatter(vecs_2d[num_cat][lang][:, 0], vecs_2d[num_cat][lang][:, 1], color=next(colors))
    # ax.scatter(vecs_2d[num_cat][lang][:, 0], vecs_2d[num_cat][lang][:, 1], vecs_2d[num_cat][lang][:, 2], color=next(colors))

plt.show()

