#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LOAD PREPROCESSED DATA
import pandas as pd
E_tokened = pd.read_json('data/D11E_tokenized_200_500.json')

# Grab a random product
# one = E_tokened[E_tokened.product_title=='Bose Soft Cover for SoundLink Mini']
import numpy as np

#only grab a product with over *n* reviews
qty = 0
while qty < 20:
    random_product = str(np.random.choice(E_tokened.product_title.unique(),1)[0])
    product_title = random_product
    one = E_tokened[E_tokened.product_title==product_title]
    qty = one.shape[0]
    print('miss: ', product_title)
print(product_title)
print('# reviews', one.shape[0])
print(one.stars.mean(), ' average stars')

#GIVEN one
print(one.iloc[0].product_title)
print('Average Stars:', one.stars.mean())
print('# reviews', one.shape[0])

# Drop product_id, product_title, it's all the same
one.drop(['product_id','product_title'], axis=1, inplace=True)

#IMPORTS
from time import time
import numpy as np
import pickle
with open('saved_data_models/D14Efull_trigrammer.pkl', mode='rb') as f:
    trigrammer = pickle.load(f)
from gensim.models import Word2Vec
w2v = Word2Vec.load('saved_data_models/D13Efull_w2v10.model')

from saved_data_models.helpers import (kw2counts, preprocess1, score_top_words, bar_plot, 
                                       simple_clusters, pd_explode, sentiment)
flatten = lambda l: [item for sublist in l for item in sublist] #a clever way to flatten a list
import spacy
nlp = spacy.load("en_core_web_sm")

t = time()
    
#REVIEWS INTO SENTENCES
f = lambda x: [sent.text for sent in nlp(x).sents]
one['sentences'] = one.review_body.apply(lambda x: f(x))
print('Reviews to sentences:', time()-t,'s')
one.head()

#EXPLODE INTO EACH ROW = EACH SENTENCE
one_sentences = pd_explode(one, 'sentences', 'sentence')
print('Explode into row=sentence:', time()-t,'s')
one_sentences.head(2)

#TOKENIZE EACH SENTENCE 
# sentence_preprocess = lambda row: [tok for tok in trigrammer[preprocess1(row.sentence)] if tok in row.tokened]
sentence_preprocess = lambda row: list(set(trigrammer[preprocess1(row.sentence)]).intersection(set(row.tokened)))
# one_sentences['sentence_tokens'] = one_sentences.sentence.apply(lambda x: trigrammer[preprocess1(x)])
one_sentences['sentence_tokens'] = one_sentences.apply(lambda row: sentence_preprocess(row), axis=1)
print('Tokenize sentences:', time()-t,'s')
one_sentences.head()

# # VALIDATION****************
# # #check if any are missing
# for ID in one_sentences.review_id.unique():
#     one_rev = one_sentences[one_sentences.review_id==ID]
#     rev_tokens = set(one_rev.tokened.iloc[0])
    
#     added = []
#     for i,row in one_rev.iterrows():
#         sent_tokens = row['sentence_tokens']
#         added.extend(sent_tokens)
        
#     added = set(added)  
#     missing = rev_tokens.difference(added)
#     if len(list(missing))>0:
#         print(rev_tokens)
#         print(missing)
        
# print('DONE!')
# # ^^^^^^^^^^^^^^^^VALIDATION



#REMOVE EXCESS SENTENCE TOKENS <---------KEEPING FOR NOW, BUT DEPRECATED BC FIXED DURING SENTENCE TOKENIZING^^
# one_sentences['sentence_tokens'] = one_sentences.apply(
#                                     lambda row: list(set(row.tokened).intersection(set(row.sentence_tokens))),
#                                         axis=1)
# # # VALIDATION - ARE THERE ANY WORDS IN SENTENCE TOKENS THAT AREN'T IN TOKENED?
# one_sentences['sentence_tokens'] = one_sentences.apply(
#                                     lambda row: print(set(row.sentence_tokens).difference(set(row.tokened))),
#                                         axis=1)
# # # ^^^ VALIDATION
# print('Remove Excess tokens:', time()-t,'s')
# one_sentences.head()

# DROP 'TOKENED', WE DON'T NEED IT ANYMORE 
one_sentences.drop('tokened',axis=1,inplace=True)

# SENTIMENT ANALYSIS
#----------->>> GO BACK AND ADJUST POLARITY SCORE
# GET polarity of each sentence #let stars skew polarity
    
one_sentences['sentence_polarity'] = one_sentences.apply(lambda row: sentiment(row.sentence, row.stars), axis=1)
print('Sentiment Analysis:', time()-t,'s')
one_sentences.head()

# #.........VALIDATION - WHAT REVIEWS ARE IN 123 STARS AND POSTIVE? 45 STARS AND NEGATIVE? NEUTRAL?
# #Negative 
# stars12 = one_sentences[(one_sentences.stars<3) & (one_sentences.sentence_polarity>0)]
# stars12

# #Positive
# stars45 = one_sentences[(one_sentences.stars>3) & (one_sentences.sentence_polarity<0)]
# stars45

# #Neutral
# stars4 = one_sentences[(one_sentences.stars==4) & (one_sentences.sentence_polarity==0)]
# stars4
# # ^^^^^^^^^^VALIDATION


# ......toks_sentDF...........FIRST BIG DF TO KEEP AND COME BACK TO
# break into each kw and each sentence
tok_sentDF = pd_explode(one_sentences, 'sentence_tokens', 'token')
print('explode into every token-sentence combo:', time()-t,'s')
tok_sentDF.head()



# CREATE POLARITY TABLE, Grouping by keyword
toksP = tok_sentDF[tok_sentDF['sentence_polarity']>0].groupby('token')                                         .agg({'review_id':list, 'sentence':list, 'stars':'count'})
toksP.rename(columns={'review_id':'p_reviews', 'sentence':'p_sentences', 
                       'stars':'p_mentions'}, inplace=True)
toksP.head()

toksN = tok_sentDF[tok_sentDF['sentence_polarity']<0].groupby('token')                                         .agg({'review_id':list, 'sentence':list, 'stars':'count'})
toksN.rename(columns={'review_id':'n_reviews', 'sentence':'n_sentences', 
                         'stars':'n_mentions'}, inplace=True)
toksN.head()

toksTOTAL = tok_sentDF.groupby('token').agg({'review_id':list, 'sentence':list, 'stars':'count'})

toksTOTAL.rename(columns={'review_id':'all_reviews','sentence':'all_sentences',
                          'stars':'all_mentions'}, inplace=True)
print('Create token-polarity subtables:', time()-t,'s')
toksTOTAL.head()

# VALIDATION toksTOTAL counts properly -- DONE



# MERG POLARITY TABLES
toks_polarity = pd.merge(toksP,toksN, how='outer', on='token')
toks_polarity = pd.merge(toks_polarity, toksTOTAL, how='outer', on='token')
toks_polarity = toks_polarity[['p_mentions','n_mentions','all_mentions','p_reviews','n_reviews','all_reviews',
                         'p_sentences','n_sentences','all_sentences']]
print('Merge polarity table:', time()-t,'s')
toks_polarity.head()

# FILL NULLS
toks_polarity[['p_mentions','n_mentions']] = toks_polarity[['p_mentions','n_mentions']].fillna(0)
toks_polarity[['p_reviews','n_reviews']] = toks_polarity[['p_reviews','n_reviews']].fillna('')
toks_polarity.p_reviews = toks_polarity.p_reviews.apply(lambda x: [] if x=='' else x)
toks_polarity.n_reviews = toks_polarity.n_reviews.apply(lambda x: [] if x=='' else x)

toks_polarity[['p_sentences','n_sentences']] = toks_polarity[['p_sentences','n_sentences']].fillna('')
toks_polarity.p_sentences = toks_polarity.p_sentences.apply(lambda x: [] if x=='' else x)
toks_polarity.n_sentences = toks_polarity.n_sentences.apply(lambda x: [] if x=='' else x)
print('Fill nulls:', time()-t,'s')
toks_polarity.head()

# ^^^^^ 2ND BIG DF TO REFERENCE

#GET KEYWORDS
one_kws = kw2counts(one.tokened)

# # ..........VALIDATION : does kw2counts match the counts in toks_polarity?
# one_kws2 = toks_polarity.sort_values('all_mentions',ascending=False)
# t1 = pd.DataFrame(one_kws, columns=['word','counts'])
# t2 = one_kws2.reset_index()[['token','all_mentions']].rename(columns={'token':'word'})
# t2 = t2.merge(t1,on='word',how='outer')
# t2['disprepency'] = t2.apply(lambda row: int(row.counts) - int(row.all_mentions), axis=1)
# t2
# # ^^^ VALIDATION^^^^^^^^

print('Get a clean set of keywords:', time()-t,'s')
one_kws.shape

# GENERATE CLUSTERS FROM KEYWORDS
clusters = simple_clusters(w2v, one_kws[:,0],choose=30, cluster_size=6, merge=0.33)
print('Cluster keywords:', time()-t,'s')
clusters

# GENERATE CLUSTER POLARITY
cluster_polarity = pd.DataFrame()
for key, words in clusters.items():
    #aggregate all values of keywords in cluster
    cluster = toks_polarity[toks_polarity.index.isin(words)].sum()
    cluster_polarity = pd.concat([cluster_polarity, cluster.T], axis=1, sort=False)

cluster_polarity = cluster_polarity.transpose()
cluster_polarity.head()
cluster_polarity['cluster_name'] = list(clusters.keys())
cluster_polarity.set_index('cluster_name', inplace=True)
print('Create cluster polarities:', time()-t,'s')
cluster_polarity.head()

# #..............VALIDATION: do counts line up?
# for i, row in cluster_polarity.iterrows():
#     if row.p_mentions + row.n_mentions > row.all_mentions:
#         print('p + n > all')
#     if len(row.p_reviews) != row.p_mentions:
#         print('p errors')
#     if len(row.n_reviews) != row.n_mentions:
#         print('n errors')
#     if len(row.all_reviews) != row.all_mentions:
#         print('all mismatch')
#         print(row.all_mentions, len(row.all_reviews))
# # ^^^^ VALIDATIONS
  

# REPRIORITIZE THE CLUSTERS
scored_clusters, _ = score_top_words(cluster_polarity, PNTcols=['p_mentions','n_mentions','all_mentions'], n=15)
print('Prioritize clusters:', time()-t,'s')
scored_clusters

# RENAME CLUSTERS
i=0
rename_dict = {}
for key, words in clusters.items():
    if i>4: break
    cluster_words = toks_polarity[toks_polarity.index.isin(words)]
    cluster_words,_ = score_top_words(cluster_words, PNTcols=['p_mentions','n_mentions','all_mentions'])
#     name_words = key.split('/')
    new_cluster_name = '/'.join([word for word in cluster_words.index[:3]])
    rename_dict[key] = new_cluster_name
scored_clusters.index = scored_clusters.reset_index().cluster_name.apply(lambda x: rename_dict[x])
scored_clusters.head()

# PLOT CLUSTERS
bar_plot(scored_clusters, PNTcols=['p_mentions','n_mentions','all_mentions'],save='D15BARPLOTTEST.png')
print('Plot clusters:', time()-t,'s')


# ...............VALIDATION - OF THE WORDS COVERED IN THE CLUSTERS, WHAT WAS LEFT OUT?
# TOTAL REVIEWS COVERAGE
all_reviews = one.review_id.unique()
covered_reviews = list(set(flatten(scored_clusters.all_reviews)))
coverage_ratio = len(covered_reviews)/len(all_reviews)
print('All reviews covered:', coverage_ratio*100)

# POLAR REVIEWS COVERAGE
all_polar_reviews = tok_sentDF[tok_sentDF.sentence_polarity!=0].review_id.unique()
covered_polar_reviews = list(set(flatten(scored_clusters.p_reviews + scored_clusters.n_reviews)))
polar_coverage_ratio = len(covered_polar_reviews)/len(all_polar_reviews)
print('Polar reviews covered:', polar_coverage_ratio*100) #NICE!

# # MISSING TERMS
# all_covered_terms = list(set(flatten(list(clusters.values()))))
# all_covered_terms

# missed_words = toks_polarity[~toks_polarity.index.isin(all_covered_terms)][['p_mentions','n_mentions','all_mentions']]
# missed_words.sort_values('all_mentions', ascending=False).head(20)
# # instructions, return, older, replacement, power, input
# missed_words.sort_values('p_mentions', ascending=False).head(20)
# # instructions, year, older, power, vision, replacement, replace
# missed_words.sort_values('n_mentions', ascending=False).head(20)
# # compatible, receive, wrong, instructions, service, connect, power, problem, problemns

# # ^^^^^^^ validation


# In[ ]:




