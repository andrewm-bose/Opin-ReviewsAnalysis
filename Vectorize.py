#!/usr/bin/env python
# coding: utf-8

# Converting tokenized reviews to word embeddings



get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('reload_ext', 'autoreload')
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
import sys
np.set_printoptions(threshold=sys.maxsize)

from time import time

from gensim.models import Word2Vec
from utils import meaning_test
                          

t=time()

# READ IN REVIEWS
print('Loading Dataset...')
E_tokened = pd.read_json('data/Electronics_tokenized.json')
print('Dataset Loaded: ', round(time()-t,2),'s')

processed_corpus = E_tokened.tokened

print('Training Word Embeddings...', round(time()-t,2),'s')

w2v = Word2Vec(processed_corpus, # input tokenized text
                 min_count=5,   # Ignore words that appear less than this
                 size=300,      # Dimensionality of word embeddings
                 workers=99,     # Number of processors (parallelisation)
                 window=6,     # Context window for words during training
                 max_vocab_size= 40*1000, #
                 compute_loss=True,
                 iter=10)       

print('Trained', round(time()-t,2),'s')

print('# reviews: ', w2v.corpus_count)
print('Total words: ', w2v.corpus_total_words)
print('Latest Training Loss: ', w2v.get_latest_training_loss())
print('Vocab size: ', len(w2v.wv.vocab))

w2v.save('models/E_w2v_gensim10.model')
print('Saved', round(time()-t,2),'s')


print('For fun, and for to test the word embeddings....')
print('Words most similar to *windows*.')
print(meaning_test(['windows'], None,  w2v))
print()
print('Words most similar to *Mac*.')
print(meaning_test(['mac'], None,  w2v))
print()
print('Words most similar to *cat*.')
print(meaning_test('cat', None, w2v))
print()
print('Words most similar to *father*.')
print(meaning_test('father', None, w2v))
print()
print('father + mother - boy')
print(meaning_test(['father','mother'],['boy'],w2v))
print()
print("Thanks :D. You're ready to run the main program!")
