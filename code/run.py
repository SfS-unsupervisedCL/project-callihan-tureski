# import sys
# sys.path.extend(['/home/ryan/Dropbox/school/usl_w18/project-callihan-mekki-tureski/code'])
import pandas as pd
from lda_model import Model
from visualization import Visualize
from data_processing import Processing, load_texts_from_directory, docs2matrix


# Replace PATH_TO_DOCUMENTS with path to documents.
PATH_TO_DOCUMENTS = ''
NUM_CATEGORIES = 30
LANGUAGE = 'english'
# LANGUAGE = 'arabic'
# LANGUAGE = 'russian'
# LANGUAGE = 'chinese'

p = Processing(stopword_lang=LANGUAGE)

# Loads .txt files from specified directory
documents = load_texts_from_directory(PATH_TO_DOCUMENTS)

# Testing on other data
# from sklearn.model_selection import train_test_split
# tsv = pd.read_csv('~/Desktop/pol_rhet_project_2017/2017_1q_data.tsv', sep="\t")
# documents = tsv['tweet']
# documents, _ = train_test_split(documents, test_size=.95, shuffle=True)

# Transforms documents into a bag-of-words model
doc_matrix, term_dictionary = docs2matrix(p.clean_docs(documents))

# Train model
lda = Model(num_categories=NUM_CATEGORIES)
ldamodel = lda.create_model(doc_matrix, term_dictionary)

# Load saved model
# ldamodel = lda.load_model(model_path="models/english_30_category_lda.model")

# Print top words for each category
for i in ldamodel.print_topics():
    for j in i:
        print(j)

# Visualize model
visualize = Visualize(num_categories=NUM_CATEGORIES)
visualize.visualize(
    ldamodel=ldamodel,
    doc_matrix=doc_matrix,
    raw_documents=documents)
