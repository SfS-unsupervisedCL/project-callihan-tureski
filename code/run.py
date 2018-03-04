# # If using the console, you may need to append the path to the code.
# import sys
# sys.path.append("C:/Users/ryanc/Dropbox/school/usl_w18/project-callihan-mekki-tureski/code")

import os
from lda_model import Model
from visualization import Visualize
from data_processing import Processing, load_texts_from_directory, docs2matrix

# Replace ROOT with path to home folder.
# ROOT = '/mnt/Shared/people/ryan/project-callihan-mekki-tureski' # sfs server
# ROOT = '/mnt/c/Users/ryanc/Dropbox/school/usl_w18/project-callihan-mekki-tureski' # Ryan linux subsystem
ROOT = 'C:/Users/ryanc/Dropbox/school/usl_w18/project-callihan-mekki-tureski' # Ryan Windows system

NUM_CATEGORIES = 50
LANGUAGES = ['english', 'chinese', 'arabic', 'russian']

for lang in LANGUAGES:

    print('Working on', lang)

    # Data processing class
    p = Processing(stopword_lang=lang)

    # Loads .txt files from specified directory
    documents, keywords, filenames = load_texts_from_directory(os.path.join(ROOT, 'un_subset', lang))

    # Separate into test and training sets, roughly 70/30
    train_idx = [i for i in range(len(documents)) if i % 3 != 0]
    test_idx = [i for i in range(len(documents)) if i % 3 == 0]

    # Training data
    train_docs = [documents[i] for i in train_idx]
    train_keywords = [keywords[i] for i in train_idx]
    train_filenames = [filenames[i] for i in train_idx]

    # Testing data
    test_docs = [documents[i] for i in test_idx]
    test_keywords = [keywords[i] for i in test_idx]
    test_filenames = [filenames[i] for i in test_idx]

    # Removed original data variables to save memory.
    del documents, keywords, filenames

    # Transforms documents into a bag-of-words matrix with term dictionary.
    train_doc_matrix, train_term_dictionary = docs2matrix(p.clean_docs(train_docs))
    test_doc_matrix, _ = docs2matrix(p.clean_docs(test_docs))

    print('Doc matrix created')

    # Train model. Comment out if unneeded
    print("Beginning training")
    lda = Model(num_categories=NUM_CATEGORIES)
    ldamodel = lda.create_model(train_doc_matrix, train_term_dictionary, language=lang)
    print('Model created')

    # # Load model if necessary
    # file = ''.join([lang, '_', str(NUM_CATEGORIES), '_category_lda.model'])
    # ldamodel = lda.load_model(model_path=os.path.join(ROOT, 'code', 'models', lang, file))


    # Displays topics with top words
    for i in ldamodel.print_topics():
        for j in i:
            print(j)

    # Cluster information to csv
    test_clusters = p.cluster_data(doc_matrix=test_doc_matrix, ldamodel=ldamodel, to_csv=True, keywords=test_keywords,
                       filenames=test_filenames, num_categories=NUM_CATEGORIES)

    # Visualize model
    visualize = Visualize(num_categories=NUM_CATEGORIES, language=lang)
    visualize.visualize(
        ldamodel=ldamodel,
        doc_matrix=test_doc_matrix,
        raw_documents=test_docs)
