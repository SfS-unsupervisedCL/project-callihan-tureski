# Latent Dirichlet Allocation Demo

## Import dependencies

```python
import os
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
```

## Corpus processing

20 Newsgroups data set. Set of 20,000 newsgroup documents spread accross 20 different topics. This makes it a nice testing ground for topic classification.

```python
remove = ('headers', 'footers', 'quotes')
newsgroups = fetch_20newsgroups(subset='all', remove=remove)
# newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)
corpus_raw = [' '.join(filter(str.isalpha, raw.lower().split())) for raw in
        newsgroups.data]
# print(newsgroups_train.data)
print("Before:\n", newsgroups.data[0])
print("After:\n", corpus_raw[0])
```


#### Output >>>
    Before:
     
    
    I am sure some bashers of Pens fans are pretty confused about the lack
    of any kind of posts about the recent Pens massacre of the Devils. Actually,
    I am  bit puzzled too and a bit relieved. However, I am going to put an end
    to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
    are killing those Devils worse than I thought. Jagr just showed you why
    he is much better than his regular season stats. He is also a lot
    fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
    fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
    regular season game.          PENS RULE!!!
    
    
    After:
     i am sure some bashers of pens fans are pretty confused about the lack of any kind of posts about the recent pens massacre of the i am bit puzzled too and a bit i am going to put an end to relief with a bit of praise for the they are killing those devils worse than i jagr just showed you why he is much better than his regular season he is also a lot fo fun to watch in the bowman should let jagr have a lot of fun in the next couple of games since the pens are going to beat the pulp out of jersey i was very disappointed not to see the islanders lose the final regular season pens

## Clean documents

```python
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 
lemmatize = WordNetLemmatizer()

def cleaning(article):
    one = " ".join([i for i in article.split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    three = " ".join(lemmatize.lemmatize(i) for i in two.split())
    four = three.split(" ")
    return four

corpus_tokenized = [cleaning(doc) for doc in corpus_raw]
print(corpus_tokenized[0])
```

#### Output >>>
    ['sure', 'bashers', 'pen', 'fan', 'pretty', 'confused', 'lack', 'kind', 'post', 'recent', 'pen', 'massacre', 'bit', 'puzzled', 'bit', 'going', 'put', 'end', 'relief', 'bit', 'praise', 'killing', 'devil', 'worse', 'jagr', 'showed', 'much', 'better', 'regular', 'season', 'also', 'lot', 'fo', 'fun', 'watch', 'bowman', 'let', 'jagr', 'lot', 'fun', 'next', 'couple', 'game', 'since', 'pen', 'going', 'beat', 'pulp', 'jersey', 'disappointed', 'see', 'islander', 'lose', 'final', 'regular', 'season', 'pen']

## Create BOW matrix for the model

```python
from time import time
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                   filename='running.log',filemode='w')

# Importing Gensim
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer


# Creating the term dictionary of our corpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
dictionary = corpora.Dictionary(corpus_tokenized)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus_tokenized]

print("Len of raw corpus: %i | Len of matrix: %i" % (len(corpus_raw), len(doc_term_matrix)))
print("Processed:\n", doc_term_matrix[0])

```

#### Output >>>
    Len of raw corpus: 18846 | Len of matrix: 18846
    Processed:
     [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 2), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 2), (18, 1), (19, 1), (20, 1), (21, 1), (22, 3), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1), (30, 2), (31, 1), (32, 1), (33, 1), (34, 1), (35, 4), (36, 1), (37, 1), (38, 1), (39, 2), (40, 1), (41, 1), (42, 1), (43, 2), (44, 2), (45, 1)]

## Train LDA model

This can take a couple minutes. I chose to use the actual number of topics for the sake of visualization.

```python
from gensim.models.ldamodel import LdaModel

start = time()

n_iter = 500
n_top_words = 5
threshold = 0.0
num_topics = 20

# Get topics
print("Actual topics:", num_topics)

# Running and Trainign LDA model on the document term matrix.
ldamodel = LdaModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary, passes=50)
print('used: {:.2f}s'.format(time()-start))

ldamodel.save('topic.model')

print("Model Saved")
```

#### Output >>>
    20
    used: 2263.27s
    Model Saved

## Load model

Load model if you already have a trained model.

```python
# Loads saved model

from gensim.models import LdaModel
ldamodel = LdaModel.load('topic.model')

print(ldamodel.print_topics(num_topics=2, num_words=4))
```

#### Output >>>
    [(3, '0.016*"file" + 0.015*"window" + 0.015*"image" + 0.011*"program"'), (0, '0.107*"x" + 0.014*"send" + 0.013*"list" + 0.011*"file"')]


## Top words in each topic

```python
# Topics
for i in ldamodel.print_topics(): 
    for j in i: print(j)
```

#### Output >>>
    0
    0.107*"x" + 0.014*"send" + 0.013*"list" + 0.011*"file" + 0.010*"mail" + 0.009*"entry" + 0.009*"email" + 0.008*"c" + 0.007*"line" + 0.007*"p"
    1
    0.032*"key" + 0.013*"encryption" + 0.013*"chip" + 0.012*"use" + 0.011*"phone" + 0.010*"new" + 0.008*"clipper" + 0.008*"security" + 0.008*"used" + 0.007*"system"
    2
    0.009*"ray" + 0.009*"adl" + 0.009*"san" + 0.009*"ca" + 0.008*"gateway" + 0.007*"burst" + 0.007*"char" + 0.007*"interactive" + 0.006*"information" + 0.006*"filled"
    3
    0.016*"file" + 0.015*"window" + 0.015*"image" + 0.011*"program" + 0.011*"use" + 0.008*"available" + 0.008*"version" + 0.007*"using" + 0.007*"software" + 0.007*"also"
    4
    0.011*"book" + 0.009*"christian" + 0.008*"one" + 0.008*"read" + 0.007*"religion" + 0.007*"article" + 0.007*"belief" + 0.007*"may" + 0.006*"atheist" + 0.006*"many"
    5
    0.018*"would" + 0.014*"people" + 0.011*"one" + 0.007*"think" + 0.007*"make" + 0.006*"even" + 0.006*"right" + 0.006*"could" + 0.006*"many" + 0.006*"like"
    6
    0.024*"armenian" + 0.011*"turkish" + 0.009*"muslim" + 0.008*"greek" + 0.006*"war" + 0.006*"russian" + 0.005*"government" + 0.005*"turk" + 0.005*"killed" + 0.005*"people"
    7
    0.015*"power" + 0.013*"water" + 0.010*"ground" + 0.009*"heat" + 0.008*"current" + 0.008*"light" + 0.006*"sound" + 0.006*"hot" + 0.006*"air" + 0.006*"circuit"
    8
    0.035*"god" + 0.015*"jesus" + 0.009*"one" + 0.009*"christian" + 0.009*"say" + 0.008*"church" + 0.007*"christ" + 0.007*"man" + 0.007*"lord" + 0.007*"u"
    9
    0.013*"drive" + 0.012*"would" + 0.010*"card" + 0.010*"please" + 0.010*"know" + 0.009*"anyone" + 0.009*"one" + 0.008*"disk" + 0.008*"system" + 0.008*"use"
    10
    0.018*"run" + 0.014*"hit" + 0.009*"player" + 0.009*"average" + 0.008*"number" + 0.008*"last" + 0.008*"pitcher" + 0.007*"base" + 0.007*"ball" + 0.007*"league"
    11
    0.010*"season" + 0.008*"period" + 0.006*"power" + 0.006*"mike" + 0.006*"division" + 0.006*"first" + 0.006*"van" + 0.005*"bos" + 0.005*"la" + 0.005*"nj"
    12
    0.044*"game" + 0.028*"team" + 0.016*"db" + 0.016*"player" + 0.014*"hockey" + 0.013*"play" + 0.011*"fan" + 0.009*"blue" + 0.008*"league" + 0.008*"nhl"
    13
    0.015*"space" + 0.007*"university" + 0.007*"research" + 0.005*"data" + 0.005*"system" + 0.005*"launch" + 0.005*"center" + 0.005*"also" + 0.005*"science" + 0.005*"information"
    14
    0.044*"car" + 0.015*"engine" + 0.014*"gm" + 0.010*"cd" + 0.009*"oil" + 0.008*"rear" + 0.008*"tire" + 0.008*"mile" + 0.007*"brake" + 0.007*"auto"
    15
    0.013*"gun" + 0.013*"drug" + 0.012*"medical" + 0.011*"health" + 0.009*"patient" + 0.008*"use" + 0.008*"number" + 0.008*"rate" + 0.007*"disease" + 0.007*"firearm"
    16
    0.018*"get" + 0.018*"would" + 0.017*"like" + 0.016*"one" + 0.012*"know" + 0.011*"think" + 0.010*"good" + 0.009*"go" + 0.009*"could" + 0.008*"time"
    17
    0.019*"israel" + 0.019*"jew" + 0.019*"israeli" + 0.016*"arab" + 0.014*"jewish" + 0.009*"palestinian" + 0.007*"lost" + 0.007*"psalm" + 0.007*"peace" + 0.007*"land"
    18
    0.012*"food" + 0.011*"insurance" + 0.007*"cover" + 0.007*"registration" + 0.007*"wiring" + 0.006*"cost" + 0.005*"plane" + 0.005*"fee" + 0.005*"price" + 0.005*"conference"
    19
    0.015*"q" + 0.010*"president" + 0.007*"going" + 0.007*"american" + 0.006*"new" + 0.006*"job" + 0.005*"state" + 0.005*"house" + 0.005*"tax" + 0.005*"press"

## Lets test it out on the test set

```python

remove = ('headers', 'footers', 'quotes')
newsgroups_test = fetch_20newsgroups(subset='all', remove=remove)
corpus_raw_test = [' '.join(filter(str.isalpha, raw.lower().split())) for raw in
        newsgroups_test.data]

print("Original Sentence:\n", newsgroups_test.data[0])

corpus_tokenized_test = [cleaning(doc) for doc in corpus_raw_test]   

doc_term_matrix_test = [dictionary.doc2bow(doc) for doc in corpus_tokenized_test]

print("\nAfter processing:\n", doc_term_matrix_test[0])
```

#### Output >>>
    Original Sentence:
     
    
    I am sure some bashers of Pens fans are pretty confused about the lack
    of any kind of posts about the recent Pens massacre of the Devils. Actually,
    I am  bit puzzled too and a bit relieved. However, I am going to put an end
    to non-PIttsburghers' relief with a bit of praise for the Pens. Man, they
    are killing those Devils worse than I thought. Jagr just showed you why
    he is much better than his regular season stats. He is also a lot
    fo fun to watch in the playoffs. Bowman should let JAgr have a lot of
    fun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final
    regular season game.          PENS RULE!!!
    
    
    
    After processing:
     [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 2), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 2), (18, 1), (19, 1), (20, 1), (21, 1), (22, 3), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1), (30, 2), (31, 1), (32, 1), (33, 1), (34, 1), (35, 4), (36, 1), (37, 1), (38, 1), (39, 2), (40, 1), (41, 1), (42, 1), (43, 2), (44, 2), (45, 1)]

## Example output for one sentence

```python
test_output = ldamodel[doc_term_matrix_test[0]]

for i in test_output:
    print(i)
```

    Output:
    (0, 0.20999999999082108)
    (1, 0.01)
    (2, 0.01)
    (3, 0.010000000072149063)
    (4, 0.010000000004415631)
    (5, 0.01000000013498671)
    (6, 0.010000000021207302)
    (7, 0.01)
    (8, 0.010000000007032847)
    (9, 0.010000001931267424)
    (10, 0.01)
    (11, 0.01)
    (12, 0.01)
    (13, 0.01)
    (14, 0.010000000113129784)
    (15, 0.01)
    (16, 0.20999999836552324)
    (17, 0.20999999994902482)
    (18, 0.01)
    (19, 0.20999999941044215)

# Visualization

## Get vectors

We have to predict the probabilities for each document and put them in a matrix.

```python
# Put probabilites into vectors
prob_matrix = np.zeros((len(doc_term_matrix_test), num_topics))

for i, doc in enumerate(doc_term_matrix_test):
    predictions = ldamodel[doc]
    idx, prob = zip(*predictions)
    prob_matrix[i, idx] = prob

```

## t-SNE

20 dimentions are hard to visualize, so lets run t-SNE to reduce the dimentionality. This can also take a couple minutes.

```python
# t-SNE

_idx = np.amax(prob_matrix, axis=1) > threshold  # idx of news that > threshold
_topics = prob_matrix[_idx]

num_example = len(_topics)

# t-SNE: 50 -> 2D
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                  init='pca')
tsne_lda = tsne_model.fit_transform(_topics[:num_example])

```

#### Output >>>
    [t-SNE] Computing 91 nearest neighbors...
    [t-SNE] Indexed 18846 samples in 0.296s...
    [t-SNE] Computed neighbors for 18846 samples in 7.465s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 18846
    [t-SNE] Computed conditional probabilities for sample 2000 / 18846
    [t-SNE] Computed conditional probabilities for sample 3000 / 18846
    [t-SNE] Computed conditional probabilities for sample 4000 / 18846
    [t-SNE] Computed conditional probabilities for sample 5000 / 18846
    [t-SNE] Computed conditional probabilities for sample 6000 / 18846
    [t-SNE] Computed conditional probabilities for sample 7000 / 18846
    [t-SNE] Computed conditional probabilities for sample 8000 / 18846
    [t-SNE] Computed conditional probabilities for sample 9000 / 18846
    [t-SNE] Computed conditional probabilities for sample 10000 / 18846
    [t-SNE] Computed conditional probabilities for sample 11000 / 18846
    [t-SNE] Computed conditional probabilities for sample 12000 / 18846
    [t-SNE] Computed conditional probabilities for sample 13000 / 18846
    [t-SNE] Computed conditional probabilities for sample 14000 / 18846
    [t-SNE] Computed conditional probabilities for sample 15000 / 18846
    [t-SNE] Computed conditional probabilities for sample 16000 / 18846
    [t-SNE] Computed conditional probabilities for sample 17000 / 18846
    [t-SNE] Computed conditional probabilities for sample 18000 / 18846
    [t-SNE] Computed conditional probabilities for sample 18846 / 18846
    [t-SNE] Mean sigma: 0.000000
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 90.214294
    [t-SNE] Error after 1000 iterations: 1.963897

## Set up metadata for visualization

```python
# find the most probable topic for each news
_lda_keys = []
for i in range(_topics.shape[0]):
    _lda_keys += _topics[i].argmax(),

# show topics and their top words
topic_summaries = []
for i in range(num_topics):
    word, _ = zip(*ldamodel.show_topic(i, topn=n_top_words))
    topic_summaries.append(' '.join(word))


# 20 colors
colormap = np.array([
  "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
  "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
  "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
])

title = "[20 newsgroups] t-SNE visualization of LDA model trained on {} news, " \
        "{} topics, thresholding at {} topic probability, {} iter ({} data " \
        "points and top {} words)".format(
  prob_matrix.shape[0], num_topics, threshold, n_iter, num_example, n_top_words)
        
```

## Visualize!

```python
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, CDSView
from bokeh.io import output_notebook
output_notebook() 

p = bp.figure(plot_width=1400, plot_height=1100,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

source = ColumnDataSource(data=dict(
  x=tsne_lda[:,0],
  y=tsne_lda[:, 1],
  color=colormap[_lda_keys][:num_example],
  content=corpus_raw_test[:num_example],
  topic_key=_lda_keys[:num_example]
  )
)

p.scatter(x='x', y='y', color='color', source=source)

topic_coord = np.empty((prob_matrix.shape[1], 2)) * np.nan
for topic_num in _lda_keys:
  if not np.isnan(topic_coord).any():
    break
  topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

# plot crucial words
for i in range(prob_matrix.shape[1]):
  p.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

# hover tools
hover = p.select(dict(type=HoverTool))
hover.tooltips = {"content": "@content - topic: @topic_key"}

# p.scatter(x=tsne_lda[:,0], y=tsne_lda[:, 1], color=colormap[_lda_keys][:num_example])
show(p)
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="9c748d1d-d4b5-47fa-80b6-7264b3e9c0c9">Loading BokehJS ...</span>
    </div>






<div class="bk-root">
    <div class="bk-plotdiv" id="c8ee93c1-1725-4548-9f73-3dd6eec677b8"></div>
</div>



