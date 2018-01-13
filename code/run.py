from code.lda_model import Model
from code.data_processing import Processing, docs2matrix, load_documents

PATH_TO_DOCUMENTS = ''
LANGUAGE = 'english'
# LANGUAGE = 'arabic'
# LANGUAGE = 'russian'
# LANGUAGE = 'chinese'

p = Processing(stopword_lang=LANGUAGE)

doc_matrix, term_dictionary = docs2matrix(p.clean_docs(load_documents(PATH_TO_DOCUMENTS)))
lda = Model()
ldamodel = lda.create_model(doc_matrix, term_dictionary, n_categories=5)

for i in ldamodel.print_topics():
    for j in i: print(j)
