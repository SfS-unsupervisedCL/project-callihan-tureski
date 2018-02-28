# import sys
# sys.path.extend(['code'])
import os
from lda_model import Model
from visualization import Visualize
from data_processing import Processing, load_texts_from_directory, docs2matrix


# Replace PATH_TO_DOCUMENTS with path to documents.
PATH_TO_CORPUS = os.path.join('..','un_subset')
NUM_CATEGORIES = 500


LANGUAGES = ['chinese', 'english', 'arabic', 'russian']
for lang in LANGUAGES:
    print('Working on', lang)
    p = Processing(stopword_lang=lang)
    # Loads .txt files from specified directory
    documents, keywords = load_texts_from_directory(os.path.join(PATH_TO_CORPUS, lang))
    documents = sorted(documents)

    train_idx = [i for i in range(len(documents)) if i % 3 != 0]
    test_idx = [i for i in range(len(documents)) if i % 3 == 0]
    train_docs = [documents[i] for i in train_idx]
    test_docs = [documents[i] for i in test_idx]
    del documents

    # Transforms documents into a bag-of-words model
    doc_matrix, term_dictionary = docs2matrix(p.clean_docs(train_docs))
    print('Doc matrix created')

    # Train model
    print("Beginning training")
    lda = Model(num_categories=NUM_CATEGORIES)
    ldamodel = lda.create_model(doc_matrix, term_dictionary, language=lang)
    print('Model created')


# Load saved model
# ldamodel = lda.load_model(model_path="models/english_30_category_lda.model")
#
# Print top words for each category
# for i in ldamodel.print_topics():
#     for j in i:
#         print(j)
#
# # Visualize model
# visualize = Visualize(num_categories=NUM_CATEGORIES)
# visualize.visualize(
#     ldamodel=ldamodel,
#     doc_matrix=doc_matrix,
#     raw_documents=documents)
