# Cross-Linguistic Topic Classification

Our goal is to test the performance of unsupervised learning (USL) techniques for topic classification on parallel documents which have been translated into languages from differing language families.  

## Motivation, method, hypotheses

Unsupervised topic classification is very well covered and there is a lot of literature out there about it. But, what we are interested in is seeing how well it works cross linguistically. Most of the literature surveyed was tried on English and we would like to see how the same model can work with languages from multiple language families. 

We would like to try a couple variations of LDA, which are standard for this kind of task as well as using a recursive K-means clustering method. The corpus we will be using is the UN parallel corpus, which has been used for many machine translation tasks in the past. It is very well organized and aligned. The corpus includes many documents which have been translated from English into: French, Spanish, Russian, Chinese, and Arabic. 

Because not all documents are translated, we will only use those documents which are available in all target languages. The XML data includes unique tags for each document along with a list of keywords which will be valuable in validating the results.

## Relevant literature 

- Dorado, Rubén, and Sylvie Ratté. "Semisupervised Text Classification Using Unsupervised Topic Information." FLAIRS Conference. 2016.
- Gowda, Harsha S., et al. "Semi-supervised Text Categorization Using Recursive K-means Clustering." International Conference on Recent Trends in Image Processing and Pattern Recognition. Springer, Singapore, 2016.
- Ko, Youngjoong, and Jungyun Seo. "Automatic text categorization by unsupervised learning." Proceedings of the 18th conference on Computational linguistics-Volume 1. Association for Computational Linguistics, 2000.
- Miller, Timothy, Dmitriy Dligach, and Guergana Savova. "Unsupervised document classification with informed topic models." Proceedings of the 15th Workshop on Biomedical Natural Language Processing. 2016.
- Rubin, Timothy N., et al. "Statistical topic models for multi-label document classification." Machine learning 88.1 (2012): 157-208.

## Available data, tools, resources

We will be using the UN parallel corpus. It includes documents of various topics in English, French, Spanish, Russian, Chinese, and Arabic. We would like to focus especially on English and Arabic, but will look at others given time.

- Ziemski, Michal, Marcin Junczys-Dowmunt, and Bruno Pouliquen. "The United Nations Parallel Corpus v1. 0." LREC. 2016.
- Website: https://conferences.unite.un.org/UNCorpus/

## Project members

- Ryan Callihan (ryancallihan)
- Nidhal Mekki (nmekki)
- Sam Tureski (porcelluscavia) 
