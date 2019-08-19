#!/usr/bin/env python
# coding: utf-8

# # Loading and preprocessing Amazon reviews

import numpy as np
import pandas as pd
import pickle

import warnings
warnings.filterwarnings('ignore')
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

from time import time

from utils import preprocess
from stopwords import get_stopwords
STOPWORDS = get_stopwords()

t=time()

# READ IN REVIEWS
print('Loading Dataset...')
reviews = pd.read_csv('data/amazon_reviews_us_Electronics_v1_00.tsv', sep='\t', error_bad_lines=False)
reviews = reviews.iloc[:1000]
print('Dataset Loaded: ', round(time()-t,2),'s')
print("Full Size:", reviews.shape[0], ' reviews')

# DROP USELESS ROWS
print('Cleaning dataframe...')
E_simple = reviews[['product_id', 'product_title', 'star_rating', 'review_headline','review_body',
                    'review_id', 'review_date']]

# DROP NULLS
print('Pre-.dropna size: ', E_simple.shape)
E_simple.dropna(inplace=True)
print("Post-.dropna size:", E_simple.shape)

# RENAME COLUMNS FOR CONVENIENCE
E_simple.rename(columns={'star_rating':'stars'}, inplace=True)
print('Clean: ', round(time()-t,2),'s')

# DROP SHORT REVIEWS, see EDA below
E_simple = E_simple[E_simple.review_body.apply(lambda x: len(x.split())>4)]
print('Dropped short reviews: ', round(time()-t,2),'s')
print('Reviews remaining: ', E_simple.shape[0])


#SAVE CHECKPOINT
print('Saving clean dataframe...')
E_simple.to_json('data/Electronics_cleaned.json')
print('Saved: ', round(time()-t,2),'s')
# SOME EDA
total_products = E_simple.product_id.unique().shape[0]
limit_extremes = E_simple.groupby('product_id').count().iloc[:,0].between(20,1000).sum()
total_reviews = E_simple.shape[0]
short_reviews = E_simple[E_simple.review_body.apply(lambda x: len(x.split())<5)].shape[0]
print('# Total products: ', total_products)
print('# Total products with 20-1000 reviews: ', limit_extremes)
print('# Total reviews: ', total_reviews)
print('# Total reviews w/ <5 words: ', short_reviews)
print(round(time()-t,2),'s')

# TOKENIZE REVIEWS
print('Tokenizing Text. Grab a coffee. This may take a while....')
processed_corpus, bigrammer, trigrammer = preprocess(E_simple.review_body, stopwords=STOPWORDS, max_gram=3)
print('Tokenized Text: ', round(time()-t,2),'s')

# SAVE TRAINED GENSIM BIGRAMMER & TRIGRAMMER MODELS
print('Saving Data & Trained Tokenizer....')
E_simple['tokened'] = processed_corpus
E_simple.to_json('data/Electronics_tokenized.json')
with open('models/bigrammer.pkl', 'wb') as f:
    pickle.dump(bigrammer,f)
with open('models/trigrammer.pkl', 'wb') as f:
    pickle.dump(trigrammer,f)
    
# SAVE A SMALLER SET FOR PRODUCTS THAT HAVE BETWEEN 20 AND 1000 REVIEWS FOR DEMO
review_counts = E_simple.groupby('product_id').count().iloc[:,0]
products20_1000 = list(review_counts[review_counts.between(20,1000)].index)
E_small = E_simple[E_simple.product_id.isin(products20_1000)]
E_small.to_json('data/Electronics_tokened_20to1000.json')
    
print('Saved', round(time()-t,2),'s')
print('Please run vectorize.py next :)')

