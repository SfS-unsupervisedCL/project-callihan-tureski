# # If using the console, you may need to append the path to the code.
# import sys
#
# sys.path.append("./code")

import os
import argparse
import logging
import sys
import datetime
import time
from lda_model import Model
from visualization import Visualize
from data_processing import Processing, load_texts_from_directory, docs2matrix


def str2bool(v):
    """
    ADAPTED FROM https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/41339640#41339640
    :param v:
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    """
    WRITTEN BY RYAN CALLIHAN
    """

    parser = argparse.ArgumentParser(description='Takes chinese texts and segments them')
    parser.add_argument('root', metavar='ROOT', type=str, nargs=1)
    parser.add_argument('--num_categories', type=int, nargs='+')
    parser.add_argument('--n_passes', type=int, nargs=1)
    parser.add_argument('--languages', metavar='LANGUAGES', type=str, nargs='+',
                        default='english chinese arabic russian')
    # parser.add_argument('--visual', type=str2bool, nargs='?', default='n', help='Visualize model. Accepts boolean '
    #                                                                             'argument')
    # parser.add_argument('--load', type=str2bool, nargs='?', default='n', help='Load pretrained model. Accepts boolean '
    #                                                                           'argument')
    # parser.add_argument('--write', type=str2bool, nargs='?', default='y', help="Writes clustering information to CSV. "
    #                                                                            "Accepts boolean argument")

    # Replace ROOT with path to home folder.
    # ROOT = '/mnt/Shared/people/ryan/project-callihan-mekki-tureski/code' # sfs server
    # ROOT = '/mnt/c/Users/ryanc/Dropbox/school/usl_w18/project-callihan-mekki-tureski' # Ryan linux subsystem
    # ROOT = 'C:/Users/ryanc/Dropbox/school/usl_w18/project-callihan-mekki-tureski' # Ryan Windows system

    ROOT = parser.parse_args().root[0]
    N_PASSES = parser.parse_args().n_passes[0]
    NUM_CATEGORIES = parser.parse_args().num_categories
    LANGUAGES = parser.parse_args().languages
    # VISUALIZE = parser.parse_args().visual
    # LOAD = parser.parse_args().load
    # WRITE = parser.parse_args().write

    ts = time.time()
    log_file = os.path.join(
        ROOT, '%s_%s_categories_%s.log' %
              ('-'.join(LANGUAGES), '-'.join([str(c) for c in NUM_CATEGORIES]),
               datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')))

    logging.basicConfig(handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    for num_categories in NUM_CATEGORIES:

        for lang in LANGUAGES:

            logging.info('Working on %s with %s categories' % (lang, str(num_categories)))

            # Data processing class
            p = Processing(stopword_lang=lang)

            # Loads .txt files from specified directory
            documents, keywords, filenames = load_texts_from_directory(os.path.join(ROOT, 'data', lang))

            # Separate into test and training sets, roughly 80/20
            document_len = len(documents)
            test_idx = list(range(document_len))[0::5]  # take every fifth
            train_idx = [i for i in range(document_len) if i not in test_idx]  # take the rest for training

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

            logging.info('Doc matrix created')

            if LOAD:
                # Load model if necessary
                lda = Model(num_categories=num_categories)
                file = ''.join([lang, '_', str(num_categories), '_category_lda.model'])
                ldamodel = lda.load_model(model_path=os.path.join(ROOT, 'models', lang, file))
            else:
                # Train model. Comment out if unneeded
                logging.info("Beginning training")
                lda = Model(num_categories=num_categories, num_passes=N_PASSES)
                ldamodel = lda.create_model(train_doc_matrix, train_term_dictionary, ROOT, language=lang)
                logging.info('Model created')

            # Displays topics with top words
            logging.info('TOP WORDS OF EACH CATEGORY FOR FINAL MODEL')
            for i in ldamodel.print_topics():
                for j in i:
                    logging.info(j)

            #
            # if WRITE:
            #     # Cluster information to csv
            #     test_clusters = p.cluster_data(doc_matrix=test_doc_matrix, ldamodel=ldamodel, to_csv=True,
            #                                    keywords=test_keywords,
            #                                    filenames=test_filenames, num_categories=num_categories)
            # if VISUALIZE:
            #     # Visualize model
            #     visualize = Visualize(num_categories=num_categories, language=lang)
            #     visualize.visualize(
            #         ldamodel=ldamodel,
            #         doc_matrix=test_doc_matrix,
            #         raw_documents=test_docs)
