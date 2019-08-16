import numpy as np
import pandas as pd
from time import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

import gensim
from gensim.models import Word2Vec, Phrases
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



# --------- PREPROCESSING --------------#
'''
Write a function to perform the pre processing steps on the entire dataset
'''

from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")


def preprocess1(text, min_tok_len = 1):
    '''
    Tokenize one document of text.
    
    text: str: text document
    min_tok_len : int : exclude all words less than this many characters.
    
    Returns: 
    list of tokens for text
    '''
    lemm_stemm = lambda tok: WordNetLemmatizer().lemmatize(tok, pos='v')
    
    result=[]
    for token in gensim.utils.simple_preprocess(text):
        if len(token) > min_tok_len:
            result.append(lemm_stemm(token))
    
    return result


def preprocess(corpus, stopwords=[], min_tok_len = 1, max_gram=2):
    '''
    Convert a series or list of texts to a list of tokens for each text item.
    
    corpus : pd.Series or list(str) : corpus of text documents to tokenize
    stopwords : list(str) : list of stopwords to remove
    min_tok_len : int : exclude all words less than this many characters.
    max_gram : int : largest number of words to combine as bi-/tri-grams.
    
    Returns : 
    list : tokenized corpus 
    gensim.Phrases : trained bigrammer
    gensim.Phrases : trained trigrammer
    '''
    first_pass = [preprocess1(doc, min_tok_len=min_tok_len) for doc in corpus]
    
    if max_gram<2: return [[tok for tok in doc if tok not in stopwords] for doc in first_pass]
    
    bigrammer = Phrases(first_pass, 
                        min_count=5, 
                        threshold=50, 
                        max_vocab_size=1*1000*1000) #10,100

    if max_gram<3:
        second_pass = bigrammer[first_pass]
    
    else:
        trigrammer = Phrases(bigrammer[bigrammer[first_pass]], 
                             min_count=20, 
                             threshold=100, 
                             max_vocab_size=1*1000*1000)
        second_pass = trigrammer[first_pass]
        
    return [[tok for tok in doc if tok not in stopwords] for doc in second_pass], bigrammer, trigrammer



def kw2counts(series):
    '''
    Converts a pandas series of tokenized text into a keyword:frequency dictionary
    '''
    import numpy as np
    import collections
    try: series = series.astype(list)
    except: pass
    all_keywords = []
    for kws in series:
        all_keywords.extend(kws)
    kw_frequencies = np.array(collections.Counter(all_keywords).most_common())
    return kw_frequencies




# ----------- WORD VECTOR ---------------#
def meaning_test2(A,B, wv1, wv2, labels=['tokened','rake'], n =5):
    sim1 = np.array(wv1.wv.most_similar(positive=A, negative=B, topn=n))
    sim2 = np.array(wv2.wv.most_similar(positive=A, negative=B, topn=n))
    df = pd.DataFrame(columns = [labels[0]+' words', labels[0], labels[1] + 'words', labels[1]])
    df[df.columns[0]] = sim1[:,0]
    df[df.columns[1]] = sim1[:,1]
    df[df.columns[2]] = sim2[:,0]
    df[df.columns[3]] = sim2[:,1]
    return df

def meaning_test(A, B, w2v, n =5):
    '''
    Check the quality of a gensim.Word2Vec model by the similar words it returns.
    
    A : list of words that contribute positively
    B : list of words that contribute negatively
    w2v : trained gensim.Word2Vec model
    n : number of similar words to return
    
    Return:
    list of n similar words and their similarities
    '''
    
    sim1 = np.array(w2v.wv.most_similar(positive=A, negative=B, topn=n))
    df = pd.DataFrame(columns = ['words', 'similarity'])
    df[df.columns[0]] = sim1[:,0]
    df[df.columns[1]] = sim1[:,1]
    return df




# -------------- SENTIMENT ----------------#
def sentiment(sentence, stars, PNthresholds = [0.24,-0.15], star_importance=0.2):
    '''
    Analyze the sentiment of a sentence using a combination of the text and the star rating
    sentence : str : sentence to analyze
    stars : int : number of stars for that review
    PNthresholds : list : threshold for what is considered a positive or negative sentiment
    star_importance : float : how much does the star rating affect sentiment classification
    
    Returns 1 if positive, 0 if neutral, -1 of negative
    '''
    vader = SentimentIntensityAnalyzer()
    pos = PNthresholds[0]
    neg = PNthresholds[1]
    
    p = vader.polarity_scores(sentence)['compound'] + (stars - 3)*star_importance
    if p>pos: return 1
    elif p< neg: return -1
    else: return 0



    
# ------------ Aggregation AND POLARITY COUNTING ---------------#
def pd_explode(df, list_col='sentences', title='sentence'):
    '''
    General purpose dataFrame exploder.
    
    df : pd.DataFrame
    list_col : column containing lists to be exploded
    title : name for new column
    
    Returns:
    
    Copy of original dataFrame, exploded so each item in each row of list_col has its own row
    '''
    #VALIDATION: number of total list_items match number of line in new df
    #   list_items in new df lineup with list_items column
    
    thing = [[[list_item] + list(row) for list_item in row[list_col]] for i,row in df.iterrows()]
    flatten = lambda l: [item for sublist in l for item in sublist] #a clever way to flatten a list
    one_row_per_item = flatten(thing)
    new_df = pd.DataFrame(one_row_per_item)
    new_df.columns = [title] + list(df.columns)
    new_df = new_df.drop(list_col,axis=1)
    return new_df


def merge1duplicate(dic, threshold):
    
    '''
    Merge TWO ONLY similar "clusters" of keywords based on their vector similarities.
    
    dict : dictionary of clusters - {'representative word': list of words in cluster}
    threshold : merging threshold for clusters, what fraction of words do they have in common
    
    Returns:
    dict : {'representative word': list of words in cluster}
    
    USES: 
    '''
        
    threshold = threshold
    duplicate_found = False
        
    for key,vals in dic.items():
        setA = set(vals)
        
        for key2,vals2 in dic.items():
            if key!=key2: #skip self
                setB = set(vals2)
                same = setA.intersection(setB)

                #merge two similar clusters
                if len(same)/len(list(setB)) > threshold:
                    #make key/key2 : vals + vals2
                    new_key = key+'/'+key2
                    new_vals = list(setA.union(setB))
                    dic[new_key] = new_vals
                    #drop both originals
                    dic.pop(key)
                    dic.pop(key2)

                    return dic, True
                
    #if no duplicates
    return dic, False

def merge_clusters(dic, threshold=0.33):
    
    '''
    Merge similar "clusters" of keywords based on their vector similarities.
    
    dict : dictionary of clusters - {'representative word': list of words in cluster}
    threshold : merging threshold for clusters, what fraction of words do they have in common
    
    Returns:
    dict : {'representative word': list of words in cluster}
    
    USES: merge1duplicate
    '''
        
    dup = True
    dic_copy = dic.copy()
    while dup:
        dic_copy, dup = merge1duplicate(dic_copy, threshold)
    return dic_copy
    
    
def simple_clusters(w2v, keywords, choose=20, cluster_size=6, merge=0.33):
    '''
    Generate "clusters" of keywords based on their vector similarities.
    
    w2v : trained gensim Word2Vec model
    keywords : list : keywords to cluster, in order of most frequent
    choose : int : number of keywords as basis for cluster centers
    cluster_size : int : initial cluster size before merging (n most similar words)
    merge : merging threshold for clusters, what fraction of words do they have in common
    
    Returns:
    dict : {'representative word': list of words in cluster}
    
    USES: merge_clusters, merge1duplicate
    '''
    #remove words that shouldn't be part of key topics
    STOPS_ = ['great', 'good', 'nice', 'perfect', 'perfectly', 'recommend', 'worth', 'work', 'br','ve']
    INCLUDES_ = ['customer','service','returns','refund','customer_service'] #prioritize these words for clustering too
    forced_keywords = [tok for tok in INCLUDES_ if tok in keywords] #make sure their in products's vocab
    keywords = [tok for tok in keywords if tok not in STOPS_] 
    
    clusters = {}
    cluster_words = list(set(keywords[:choose] + forced_keywords)) # top *choose* product keywords + forced_keywords
    for tok in cluster_words:
        
        try: w2v.wv.vocab[tok] #skip OOV words
        except: continue
        
        qty = 0
        nn = cluster_size + 20 
        exit = False
        timeout = 0
        threshold = 0.5
        while qty < cluster_size and not exit and timeout < 5:
            
            #get most similar words (get some extra)
            similars = np.array(w2v.wv.most_similar(tok,topn=nn))
            
            # drop words that are too different
            most = np.max(similars[:,1].astype(float))
            similars = [tup[0] for tup in similars if float(tup[1])/most > threshold]
            if len(similars) < cluster_size: exit=True #if we've already reached all similar words, don't try again  
                  
            #prune down OOV words
            similars = [tok for tok in similars if tok in keywords][:cluster_size] 
            qty = len(similars) #check if correct quantity is met
            nn += 20 #grab more next time if not enough
            timeout += 1 

        #add self to list of similar words
        similars.append(tok)
        clusters[tok] = similars
    
    clusters = merge_clusters(clusters, merge)
                
    return clusters



def score_top_words(Table, n=20, PNTcols=['p_mentions','n_mentions','all_mentions'], 
                                            PNTpriority=(1.5,2,1,1,1), aslist=False):
    '''
    Take in a table of sentiment counts and rearrange them by selection priority.
    
    Table : pd.DataFrame : of from index = word, with cols = counts of word occurence
    n : int : number of output words.
    PNTcols : list : column headers for positive, negative, and total counts
    PNTpriority : tuple : weights that prioritize (respectively): 
                            #positive counts, #negative counts, #total counts, pos/neg ratio, neg/pos ratio
    
    Returns:
    keeps : pd.DataFrame : re-sorted by selection priority of length n
    drops : pd.DataFrame : all words not kept
    if aslist: return list of words
    '''
    p_col = PNTcols[0]
    n_col = PNTcols[1]
    t_col = PNTcols[2]
    
    Table = Table.copy()
        
    #Drop warnings
    pd.set_option('mode.chained_assignment', None)
    
    #Generate pos/neg ratios, avoiding /0 issues
    ratio = lambda x,y: x/y if y > 0 else x**0.5 #choose a non-drastic ratio when y is 0
    Table['p_ratio'] = Table.apply(lambda row: ratio(row[p_col],row[n_col]), axis=1) 
    Table['n_ratio'] = Table.apply(lambda row: ratio(row[n_col],row[p_col]), axis=1) 

    #Get the max of each of our subscores for scaling
    max_p = max(Table[p_col].max(), 1) #if no positive mentions, set to 1 to avoid div by 0 later
    max_n = max(Table[n_col].max(), 1) #if no negative mentions, set to 1 to avoid div by 0 later
    max_total = Table[t_col].max()
    max_p_ratio = max(Table.p_ratio.max(), 0.01) #if no positive mentions, set to 1 to avoid div by 0 later
    max_n_ratio = max(Table.n_ratio.max(), 0.01) #if no negative mentions, set to 1 to avoid div by 0 later
    # in the 'div by 0' cases, that subscore = 0 because max for that feature = 0 (eg N*row[n_col]/max_n > 2*0/0.01 =0)
    
     #Create selection score
    P = PNTpriority[0]
    N =PNTpriority[1]
    T = PNTpriority[2]
    Pr = PNTpriority[3]
    Nr = PNTpriority[4]

    Table['score'] = Table.apply(lambda row: P*row[p_col]/max_p +
                                           N*row[n_col]/max_n +
                                           T*row[t_col]/max_total + 
                                           Pr*row.p_ratio/max_p_ratio + 
                                           Nr*row.n_ratio/max_n_ratio, 
                                        axis=1)
    
    #Output df or list of kept/dropped words
    keeps = Table.sort_values('score', ascending=False).head(n)
    drops = Table.sort_values('score', ascending=False).tail(Table.shape[0] - n)
    
    if aslist: return list(keeps.index), list(drops.index)
    else: return keeps, drops

    


# ------------ Results -----------------#
def bar_plot(DF, PNTcols = ['p_mentions','n_mentions','all_mentions'], title=None, save=None):
    '''
    Converts a sentiment count table to a bar plot.
    
    DF : DF of the form index:word, cols=['positive counts','negative counts','total counts']
    
    Returns: None
    To try stacked barplots: https://python-graph-gallery.com/12-stacked-barplot-with-matplotlib/
    '''

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="whitegrid")
    
    pos = PNTcols[0]
    neg = PNTcols[1]
    total = PNTcols[2]
    
    #copy, sort, reset index
    Table = DF.copy()
    Table = Table.sort_values(by=total, ascending=False).reset_index()
    index = Table.columns[0] #get column name of former index
    
    # Scale the data
    scale = lambda x: np.log(x+3)**3 -1.1 if x != 0 else 0
    Table['p_scaled'] = Table[pos].apply(lambda x: scale(x))
    Table['n_scaled'] = Table[neg].apply(lambda x: scale(x))
    Table['total_scaled'] = Table[total].apply(lambda x: scale(x))

    # Initialize the matplotlib figure
    fig_size = (10,15)
    fig, ax = plt.subplots(figsize=fig_size)

    # # Plot the total counts
    sns.set_color_codes("pastel")
    T = sns.barplot(x="total_scaled", y=Table[index], data=Table,
                label="Total", color="b")

#     # Plot Positive counts
    sns.set_color_codes("bright")
    P = sns.barplot(x="p_scaled", y=Table[index], data=Table,
                label="pos mentions", color="g")

    # Plot negative counts
    sns.set_color_codes("bright")
    N = sns.barplot(x="n_scaled", y=Table[index], data=Table,
                label="neg mentions", color="r")
    
    # Set labels for bars
    for i, row in Table.iterrows():
        if row['total_scaled'] - row['p_scaled'] <13: #if total and positive numbers overlap, shift total
            T.text(row['total_scaled'], row.name - 0.2, row[total], color='blue', ha="left", weight='bold')
        else:             
            T.text(row['total_scaled']-1, row.name - 0.2, row[total], color='blue', ha="right", weight='bold')
        if row['p_scaled'] < 7: # if total positive is very small, move away from the y-axis
            P.text(row['p_scaled']+1, row.name - 0.2, int(row[pos]), color='#046302', ha="right", weight='bold')
        else: 
            P.text(row['p_scaled']-1, row.name - 0.2, int(row[pos]), color='#046302', ha="right", weight='bold')

        if row['n_scaled'] < 7:# if total negative is very small, move away from the y-axis
            N.text(row['n_scaled'], row.name + 0.35, int(row[neg]), color='#630202', ha="left", weight='bold')
        else: 
            N.text(row['n_scaled']-1, row.name + 0.35, int(row[neg]), color='#630202', ha="right", weight='bold')
    
    if title: plt.title(title, )
    if save: plt.savefig(save)
    plt.xlabel('') #remove xlabel
    plt.ylabel('') #remove ylabel
    plt.show()
    plt.close() # save memory
    
    
    
def cluster_graph(cluster_dict, title=None, save=None):
    '''
    Graphs cluster keywords with children.
    cluster_dict: dict: of from {name:list(of words as str)}
    title: str
    save: str: save location if any

    Returns: None
    '''
    flatten = lambda l: [item for sublist in l for item in sublist] #a clever way to flatten a list

    s = cluster_dict.copy()
    for old_key in list(s.keys()):
        new_key = old_key.replace('/','\n',9)
        s[new_key] = s.pop(old_key)

    centroids = [key for key in s.keys()]
    subwords = flatten([word for word in s.values()])

    fig, ax = plt.subplots(figsize=(20,20))
    G = nx.from_dict_of_lists(s)

    #Centroids - 
    nx.draw_networkx_nodes(G, pos=graphviz_layout(G), nodelist=centroids, node_color='r',
                           node_size=2000, node_shape = 'o', alpha=1) # ‘so^>v<dph8’ (default=’o’)

    #subwords - 
    nx.draw_networkx_nodes(G, pos=graphviz_layout(G), nodelist=subwords, node_color='lightblue', 
                           node_size=400, node_shape = 'o', alpha=1) # ‘so^>v<dph8’ (default=’o’)  

    #edges - 2 shades for effect
    nx.draw_networkx_edges(G, pos=graphviz_layout(G), with_labels=True,
                           width=1, alpha=1, edge_color='r')
    nx.draw_networkx_edges(G, pos=graphviz_layout(G), with_labels=True,
                           width=7, alpha=0.5, edge_color='r')
    #labels
    nx.draw_networkx_labels(G, pos=graphviz_layout(G), labels={w:w for w in centroids}, 
                            font_size=14, font_weight='bold')
    nx.draw_networkx_labels(G, pos=graphviz_layout(G), labels={w:w for w in subwords}, 
                            font_size=10)

    if title: plt.suptitle(title,size=30,y=.87)
    if save: plt.savefig(save)
    plt.draw()
    plt.show()
    plt.close() #save memory
    
    

    
    
    
    
    
    
    
    

    
    
    
    
    
# ................. NOT USED - FROM OLDER VERSIONS ....................#


# >>>>>>> NOT USED - WORD2Vec
def terms2vec(w2v, kws):
    '''
    Generate get word embedding subset of a word2vec model from keywords
    
    w2v : gensim Word2Vec : vectorspace these words exist in
    kws : list/array/series : words for our product subset
    
    Returns:
    X : word embeddings subset
    kws_idx  : (word, index) pair for all keywords
    '''
    # get vocab of full space in order
    ordered_keys = list(w2v.wv.vocab.keys()) 
    
    #convert to word , index array
    ordered_keys = np.array([(word,i) for word,i in enumerate(ordered_keys)])

    # Need (word,idx) for every word in individual keywords
    kws_idx = np.array([tup for tup in ordered_keys if (tup[1] in list(kws))])

    # create embedding submatrix on just terms in our search space
    X = np.array([w2v.wv[tup[1]] for tup in kws_idx]) 
    return X, kws_idx


# ...... NOT USED >>>>>>> KMEANS CLUSTERING

# -------------- KMeans Topic Modeling --------------#
def central_terms(embeddings, KM=None, n=5, keys = [], w2v=None, k=15):
    '''
    Get the central terms to each topic cluster
    
    embeddings : matrix (words x latents)
    KM : sklearn KMeans model
    n : number of top terms
    keys : sorted name labels for word embeddings
    w2v : word2vec system for similarity calculations
    
    Returns: 
    pd.DataFrame of top words for each topic
    KMeans losses
    trained KMeans model
    '''
    
    print('Transforming Embeddings')
    if KM == None:
        KM = KMeans(
            n_clusters=k,       
            init='k-means++',
            n_init=12,        
            max_iter=100, 
            tol=0.001,  
            precompute_distances='auto',
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs=-1,
            algorithm='auto',
            )
    #Override stated k if KM is trained
    k = KM.n_clusters
    
    # Fit model and transform embeddings into distances from each cluster
    distances = KM.fit_transform(embeddings)
    losses = KM.inertia_ #score(X)
    
    tups = [] #len = #words, form (word_idx, distance to it's center)
    for i, topic in enumerate(KM.labels_): #KM.labels_ = cluster classification for each word
        tup = topic, distances[i,topic] #get distance from each word to it's closest center. 
        tups.append(tup)
    df = pd.DataFrame(tups, columns=['topic','distance'])
    
    print('Getting top {} words per {} topics'.format(n, k))
    if len(keys) >0:
        df['token'] = keys
        
        #GENERATE A TABLE OF THE n WORDS CLOSEST TO EACH CENTROID
        table = pd.DataFrame()
        for t in df.topic.unique(): #for each KM cluster...
            df_temp = df[df.topic == t].sort_values(by='distance').head(n) #get n closest words
            if w2v != None: 
                
                #generate a second list of words that are most similar from w2v
                qty = 0
                nn = n + 10
                while qty < n: #ensure most_similar words are in product's vocab
                    commons = np.array(w2v.wv.most_similar(df_temp.token,topn=nn))[:,0] #get extra words
                    commons = [tok for tok in commons if tok in keys][:n] #prune down OOV words
                    qty = len(commons) #check if correct quantity is met
                    nn += 10 #grab more next time if not enough
                
                # THIS CAN BE MADE MORE EFFICIENT BY MERGING WITH ABOVE  
                # sometimes a topic will have fewer words, in that case, match size
                if df_temp.shape[0] < n: commons = commons[:df_temp.shape[0]]
                df_temp['most_common'] = commons 
            table = pd.concat([table, df_temp])
            
    else:
        pass #potential implementation without keys
    return table.sort_values('topic') ,losses, KM 


def reviews2topics(df, w2v, n=6, KM=None, k=11, search_col='tokened', Time=True):
    '''
    Converts a tokenized corpus into to list topics defined by categories.
    
    df : pd.DataFrame that contains tokenized column
    w2v : trained w2v vectorizor
    KM : trained KMeans model
    n : number of top words to return
    k : number of topics to generate (if KM == None)
    search_col : column in df to search for tokens
    Time: print time bool
    
    Returns:
    pd.DataFrame of topics
    loss : float : Sum of Distances to centroids
    KM : trained KMeans model
    
    USES: kw2counts, terms2vec, central_terms
    '''
    
    t = time()
    
    #get key words from corpus
    print('Getting keywords...')
    kws = kw2counts(df[search_col])[:,0] 
    
    #create the word vectors
    print('Vectorizing keywords...')
    X, X_labels = terms2vec(w2v, kws)

    #create a KMeans model for each k
    if KM == None:
        KM = KMeans(
            n_clusters=k,     
            init='k-means++',
            n_init=12,        
            max_iter=100, 
            tol=0.001,  
            precompute_distances='auto',
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs=-1,
            algorithm='auto',
            )
    
    print('Generating Topics....')
    df_topics, losses, KM2 = central_terms(X, KM=KM, n=n, keys=X_labels[:,1], w2v=w2v); 
                                                            #X_labels is of form [[idx,word],...]
    if Time: print(time()-t,'s')
    return df_topics, losses, KM2


## This is very redundant compared to central terms
# def reviews2topics(w2v, keys, KM=None, k=15, Time=True):
#     '''
#     Converts Keywords to list topics defined by categories.
    
#     w2v : trained w2v vectorizor
#     keys : list of keywords that match word vectors
#     KM : sklearn KMeans model (else created by default)
#     Time: print time bool
    
#     Returns:
#     pd.DataFrame of topics
#     loss : float : Sum of Distances to centroids
#     KM : trained KMeans model
    
#     USES: terms2vec, central_terms
#     '''
#     t = time()
#     #create the word vectors
#     X, X_labels = terms2vec(w2v, keys)

#     #create a KMeans model for each k
#     if KM == None:
#         KM = KMeans(
#             n_clusters=k,     
#             init='k-means++',
#             n_init=12,        
#             max_iter=100, 
#             tol=0.001,  
#             precompute_distances='auto',
#             verbose=0,
#             random_state=None,
#             copy_x=True,
#             n_jobs=-1,
#             algorithm='auto',
#             )
    
#     KM.fit_transform(X) #make sure KM is fitted for X, even though central terms does it too
        
#     losses = KM.inertia_#score(X)
#     df_topics = central_terms(X, KM=KM, n=8, keys=X_labels[:,1], w2v=w2v, k=k); #X_labels is of form [[idx,word],...]
#     if Time: print(time()-t,'s')
#     return df_topics, losses, KM


def ksearch2topics(w2v, keys, krange, topN = 5, Time=True):
    '''
    Generate lists of topic dataframes and trained KMeans models for each value for k in a gridsearch
    
    w2v : trained w2v vectorizor
    keys : list of keywords that match word vectors
    Time: print time bool
    
    Returns:
    pd.DataFrame of topics
    loss : float : Sum of Distances to centroids
    KM : trained KMeans model
    
    USES: central_terms, terms2vec, central_terms
    '''
    t0 = time()
    
    #create the word vectors
    print('Vectorizing keywords...')
    X, X_labels = terms2vec(w2v, keys)
    
    
    topics_dfs = []
    losses_lst = []
    models = []
    print('Gridsearch...')
    for i,k in enumerate(krange):
        print(i)
        
     #create a KMeans model for each k
        KM = KMeans(
            n_clusters=k,     
            init='k-means++',
            n_init=12,        
            max_iter=100, 
            tol=0.001,  
            precompute_distances='auto',
            verbose=0,
            random_state=None,
            copy_x=True,
            n_jobs=None,
            algorithm='auto',
            )
        
        try:
            topics, losses, KM2 = central_terms(X, KM, topN, X_labels[:,1], w2v=w2v, k=k)
        except: #sometimes the above fails, need to debug later
            print(i, 'error')
            topics = pd.DataFrame({'col1': ['err', 'err'], 'col2': ['err', 'err']})
            losses = 0
            models = None
            
        topics_dfs.append(topics) 
        losses_lst.append(losses)
        models.append(KM2)
        
    if Time: print(time()-t0,'s')
    return topics_dfs, losses_lst, models, krange


# ...... NOT USED >>>>>>> PCA AND VISUALIZATION


def table2topNwords(DF, topN):
    '''
    From a pd.DataFrame, get top words for each topics
    
    df : pd.DataFrame : columns=['topic', 'distance', 'token', 'most_common']
    topN : int : max words per topic to extract
    
    Returns:
    topN : list of topN*2 words for all categories
    top2 : list of top 2 words for all categories
    k : number of topics
    '''
    df = DF.copy()
    topNs = set()
    top2s = set()
    for t in df.topic.unique():
        temp = df[df.topic ==t]
        topNs.update(list(temp.token.iloc[:topN]) + list(temp.most_common.iloc[:topN]))
        top2s.update([temp.token.iloc[0] , temp.most_common.iloc[0]])
    k= len(df.topic.unique())
    return topNs, top2s, k


def generate_colors(n=20, cmap='nipy_spectral'):
    '''
    n : length of array
    map : str : matplotlib color style (suggested alternative: 'prism')
    
    Returns: list of unique colors for plotting
    '''
    cm = plt.get_cmap(cmap)
    colors = np.array([cm(1.*i/n) for i in range(n)])
    
    #Darken light colors
    adjusted_colors = []
    for c in colors:
        if sum(c) > 2.1:
            adjusted_colors.append([c[0]*.7, c[1]*.7, c[2]*0.7, 1])
        else:
            adjusted_colors.append(c)
    return np.array(adjusted_colors)


def table2topBOW(DF, n):
    '''
    Convert a pd.DataFrame to list of topN words and top2 words for each topic
    '''
    df = DF.copy()
    m= int(n/2)
    bowN = set()
    bow2 = set()
    for t in df.topic.unique():
        temp = df[df.topic==t]
        bowN.update(list(temp.token.iloc[:m]) + list(temp.token.iloc[:m]))
        bow2.update( [temp.token.iloc[0] , temp.token.iloc[0]])
    return list(bowN), list(bow2)



def wvPCA(X, X_labels, KM, topNwords, PCA_type='PCA',topwords = [] , save=None, title=None, subtitle=None, 
          plot=False, clean=False):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    '''
    Plots a PCA plot from embeddings matrix and KM clusters
    
    X : word vector embeddings matrix (words x latents)
    X_labels : list : word names for vectors 
    KM : trained sklearn kmeans model
    PCA_type: type of PCA to do - sklearn PCA vs TSNE 
    topNwords : top n words for each topic to highlight
    topwords : top words for all topics to highlight
    save : filepath to save to
    
    Returns : trained PCA/TNSE Model
    '''
    
    t = time()
    wvs = X #copy X
    
    #combine word vectors and centroids for transforming
    wvs_centroids = np.vstack((wvs, KM.cluster_centers_))
    
    #CHOOSE PCA TYPE
    if PCA_type.lower()=='tsne':
        pca = TSNE(n_components=3, n_iter=2000, perplexity=10)
    elif PCA_type.lower()=='pca':
        pca = PCA(n_components=2 )
    else: return 'No PCA Model Recognized, choose pca or tsne'
    
    # Transform word vectors into 2 dimensions
    print('Projecting Word Vectors...')
    np.set_printoptions(suppress=True)
    P = pca.fit_transform(wvs_centroids) #wvs and centroids
    
    #retrieve X
    XP = P[:len(wvs),:]
    #retrieve centroids
    CP = P[len(X):,:]
    
    labels = X_labels #words
    centroid_labels = ['C'+str(i) for i in range(20)] #C1, C2, C3, ....

    print('{} Fitted'.format(PCA_type.upper()))

    #GENERATE KMEANS COLOR MAPPING
    colors = generate_colors(KM.n_clusters)
    c_lst = [colors[cluster] for cluster in KM.labels_]
    centroid_c_lst = colors[range(CP.shape[0])]

    # ATTEMPT 3D
    # plt.figure(figsize=(18, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(P[:, 0], P[:, 1], P[:,2], c=c_lst, edgecolors='r') 
    # for label, x, y,z in zip(labels, T[:, 0], T[:, 1], T[:,2]):
    #     ax.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

#     Else 2D
    fig, ax = plt.subplots(1, figsize=(18, 10))
    ax.scatter(XP[:, 0], XP[:, 1], c=c_lst, edgecolors='g', marker='.', alpha=.9)

    i = 0 #setting up an index to avoid complicated forloop init
    for label, x, y in zip(labels, XP[:, 0], XP[:, 1]):
        
        #Set color for current data point by cluster
        c = c_lst[i] 
        
        # Stylize the top 2 words for each topic
        if label in topwords:
            ax.annotate(label, 
                         xy=(x+0.06, y+0.03), 
                         xytext=(0, 0), 
                         textcoords='offset points',
                         **{'fontsize':'large', 'fontweight':'bold', 'alpha':1, 'color':c})
        
        # Stylize the top n words for each topic
        elif label in topNwords:
            ax.annotate(label, 
                         xy=(x+0.06, y+0.03), 
                         xytext=(0, 0), 
                         textcoords='offset points',
                         **{'fontsize':'small', 'fontweight':'bold', 'alpha':1, 'color':c})
        else:
            if clean: alpha = 0
            else: alpha = 0.3
            ax.annotate(label, 
                         xy=(x+0.06, y+0.03), 
                         xytext=(0, 0), 
                         textcoords='offset points',
                         **{'fontsize':'small', 'fontweight':'light', 'alpha':alpha})
        i += 1 
        
    #Plot centroids
    ax.scatter(CP[:, 0], CP[:, 1], c=centroid_c_lst, edgecolors='g')
    i = 0 #setting up an index to avoid complicated forloop init
    for label, x, y in zip(centroid_labels, CP[:, 0], CP[:, 1]):
        c = centroid_c_lst[i]
        ax.annotate(label, 
                     xy=(x+0.06, y+0.03), 
                     xytext=(0, 0), 
                     textcoords='offset points', 
                     **{'fontsize':'large', 'fontweight':'bold', 'color':c})
        i += 1

    
    #Plot the title
    if title == None: 
        title = "{} on {} Kmeans Clusters (Centroids eg 'C1,C2,...')".format(PCA_type.upper(), int(KM.n_clusters))
    ax.set_title(title, fontdict={'fontsize':'xx-large', 'fontweight':'bold'})
    
    if subtitle == None: 
        subtitle = "Top words for each cluster colored and bolded"
    fig.suptitle(subtitle, y=.88, fontsize=18)
    if save: fig.savefig(save, format='png')
        
    print(time()-t,'s')
    if plot: fig.tight_layout()
    if plot: fig.show()
    plt.close(fig)
    
    return pca, fig, ax



# ...... NOT USED >>>>>>> AGREGATION AND POLARITY
def polarity_counter(series, kws, updown):
    '''
    Searches a pd.Series of tokenized text for keywords, and products a table of those counts with *update* as the column title
    
    Recommended use: polarity_counter(positive_reviews, kws, 'up')
                     polarity_counter(negative_reviews, kws, 'down')
                     polarity_counter(all_reviews, kws, 'total')
    
    Parameters:
    series: pd.Series: of tokenized text to search
    kws: list: of unique key words to search for 
        OR dict : clusters of words of form {'representative term':list of words}
    updown: str: name of column to generate
    
    Returns:
    pd.DataFrame of form with words as indices and one column as the counts
    '''
    updown = updown.lower()
    kw_dict = {}
    for revtoks in series: #for the tokens in each review
        
        # if tallying clusters of words
        if type(kws) ==dict:

            for key,words in kws.items(): # for every cluster
                if set(words).intersection(set(revtoks)): #if cluster intersects with review

                    #if kw already in dict
                    try: kw_dict.update(
                                        {key : {updown:kw_dict[key][updown]+1}}
                                        )
                    #if kw added for first time
                    except: 
                        kw_dict.update({key:{updown:1}})

        # if tallying just keywords   
        else:
            for word in set(kws).intersection(set(revtoks)): #for each keyword found in that review

                #if kw already in dict
                try: kw_dict.update(
                                    {word : {updown:kw_dict[word][updown]+1}}
                                    )
                #if kw added for first time
                except: 
                    kw_dict.update({word:{updown:1}})
                
    return pd.DataFrame.from_dict(kw_dict, orient='index')


def build_topWords_table(oneDF, kws, ratings_col ='stars', search_col='tokened'):
    '''
    Builds a table of words and sentiment counts on a PER REVIEW LEVEL basis
    oneDF: pd.DataFrame:  of reviews for one product, with columns contained tokenized text and stars ratings
    kws: list: unique keywords to search for in text
        OR dict : clusters of words of form {'representative term':list of words}
    ratings_col: name of column in dataFrame to get ratings
    search_col: name of column in dataFrame to get tokenized reviews
    
    
    Returns:
    pd.DataFrame of (top words for one produce) with counts positive/negative/total mentions
    
    USES: polarity_counter
    '''
        
    one_ups = oneDF[oneDF[ratings_col] > 4]
    one_dns = oneDF[oneDF[ratings_col] < 3]
    table = polarity_counter(one_ups[search_col], kws, 'up')
    table = table.join(
                        polarity_counter(one_dns[search_col], kws, 'down')) \
                        .fillna(0).astype(int)
    
    table = table.join(
                        polarity_counter(oneDF[search_col],kws, 'total') \
                        .fillna(0).astype(int)
                      )
    
    if type(kws) == dict: 
        table['cluster'] = [kws[key] for key in table.index]

    return table


def subcluster_polarity(product_df, clusters_dict, ratings_col ='stars', search_col='tokened'):
    '''
    Get the polarity distribution for each word in a set of word-clusters
    
    product_df : pd.DataFrame : one product with ratings and tokenized reviews
    cluster_dict : dictionary : dictionary of form {'cluster keyword': list of words}
    ratings_col : column in product_df that contains ratings
    search_col : column in product_df that contains tokenized reviews
    
    Returns: 
    pd.DataFrame of words and their polarity.
    
    USES:
    build_topWords_table
    
    '''
    
    df = pd.DataFrame()
    for key, words in clusters_dict.items():
        df_temp = build_topWords_table(product_df, words, ratings_col =ratings_col, search_col=search_col)
        df_temp['cluster_name'] = key
        df = pd.concat([df, df_temp], sort=False)
    
    return df.sort_values('cluster_name')


def rename_clusters(cluster_DF, subcluster_DF):
    '''
    Reprioritize the names of clusters based on which words are they highest selection priority in that cluster.
    Also fixes issues with names.
    
    cluster_DF: pd.DataFrame : Sentiment Table of clusters
    subcluster_DF : pd.DataFrame : Sentiment Table for each word in cluster
    
    Returns:
    renamed_cluster_DF : cluster_DF with clusters renamed
    renamed_subcluster_DF : subcluster_DF with clusters renamed
    old2new_dict : dictionary used to rename clusters {old name:new name}
    
    USES: score_top_words, 
    '''
    cluster_df = cluster_DF.copy()
    subcluster_df = subcluster_DF.copy()
    
    old2new_dict = {}
    # Get naming keyword priorities
    renamed_subcluster_df = pd.DataFrame()
    for cluster_name in subcluster_df.cluster_name.unique():
        df_temp = subcluster_df[subcluster_df.cluster_name==cluster_name]
        df_temp,_ = score_top_words(df_temp, n=20, PNTcols = ['up','down','total'], 
                                    PNTpriority=(1.5,2,1,1,1), aslist=False)
        
        #get new cluster name
        old_cluster_name = cluster_name #save for generating old2new dict
        cluster_name_words = cluster_name.split('/')
        top_words = list(df_temp.index[:2])
        for word in top_words[::-1]: #insert top words in backwards
            if word in cluster_name_words:
                cluster_name_words.remove(word) #remove the word if already present
            cluster_name_words = [word] + cluster_name_words #add it back to the front
        new_cluster_name = '/'.join([word for word in cluster_name_words[:3]])
        
        #rename the cluster in subcluster_df
        df_temp['cluster_name'] = new_cluster_name
        renamed_subcluster_df = pd.concat([renamed_subcluster_df, df_temp])
        
        old2new_dict[old_cluster_name] = new_cluster_name
    
    cluster_df['name'] = list(cluster_df.reset_index()['index'].apply(lambda x: old2new_dict[x]))
    renamed_cluster_df = cluster_df.set_index('name')

    return renamed_cluster_df, renamed_subcluster_df, old2new_dict   
    
    
    
    
