import numpy as np
import pandas as pd
from time import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use('seaborn')



# --------- PREPROCESSING --------------#
'''
Write a function to perform the pre processing steps on the entire dataset
'''

from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    '''
    nltk lemmatizer
    '''
    return WordNetLemmatizer().lemmatize(text, pos='v')

# Tokenize and lemmatize
def preprocess(text, stopwords=[], min_tok_len = 3):
    '''
    Body of Text to tokens list
    '''
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in stopwords and len(token) > min_tok_len:
            result.append(lemmatize_stemming(token))
            
    return result

def kw2counts(series):
    '''
    Converts a pandas series into a keyword:frequency dictionary
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
def meaning_test2(A,B, wv1, wv2, labels=['gensim','rake'], n =5):
    sim1 = np.array(wv1.wv.most_similar(positive=A, negative=B, topn=n))
    sim2 = np.array(wv2.wv.most_similar(positive=A, negative=B, topn=n))
    df = pd.DataFrame(columns = [labels[0]+' words', labels[0], labels[1] + 'words', labels[1]])
    df[df.columns[0]] = sim1[:,0]
    df[df.columns[1]] = sim1[:,1]
    df[df.columns[2]] = sim2[:,0]
    df[df.columns[3]] = sim2[:,1]
    return df

def meaning_test(A,B, wv1, labels=['gensim'], n =5):
    sim1 = np.array(wv1.wv.most_similar(positive=A, negative=B, topn=n))
    df = pd.DataFrame(columns = [labels[0]+' words', labels[0]])
    df[df.columns[0]] = sim1[:,0]
    df[df.columns[1]] = sim1[:,1]
    return df

def terms2vec(w2v, kws):
    '''
    Generate get word embedding subset of a word2vec model from keywords
    
    w2v : gensim Word2Vec : vectorspace these words exist in
    kws : list/array/series : words for our product subset
    
    Returns:
    X : word embeddings subset
      : (word, index) pare for all keywords
    '''
    # get vocab of full space in order
    ordered_keys = list(w2v.wv.vocab.keys()) 
    
    #convert to word , index array
    ordered_keys = np.array([(word,i) for word,i in enumerate(ordered_keys)])

    # Need (word,idx) for every word in individual keywords
    # I feel like this is backwards, maybe rewrite this to go faster
    kws_idx = np.array([e for e in ordered_keys if (e[1] in list(kws))])

    # create embedding submatrix on just terms in our search space
    X = np.array([w2v.wv[e[1]] for e in kws_idx]) 
    keywords = kws_idx
    return X, keywords

# -------------- KMeans Topic Modeling --------------#
def central_terms(embeddings, KM=None, n=5, keys = [], w2v=None, k=15):
    '''
    Get the central terms to each topic cluster
    
    embeddings : matrix (words x latents)
    KM : sklearn KMeans model
    n : number of top terms
    keys : sorted name labels for word embeddings
    w2v : word2vec system
    
    Returns: 
    pd.DataFrame of top words for each topic
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
            n_jobs=None,
            algorithm='auto',
            )
    
    distances = KM.fit_transform(embeddings)
    tups = []
    for i, topic in enumerate(KM.labels_):
        tup = topic, distances[i,topic] #get distance from each word to it's closest center.
        tups.append(tup)
    df = pd.DataFrame(tups, columns=['topic','distance'])
    
    print('Getting top {} words per {} topics'.format(n, k))
    if len(keys) >0:
        df['token'] = keys
        
        #GENERATE AT TABLE OF THE n WORDS CLOSEST TO EACH CENTROID
        table = pd.DataFrame()
        for t in df.topic.unique():
            df_temp = df[df.topic == t].sort_values(by='distance').head(n)
            if w2v != None: 
                #generate a second list of words that are most similar from w2v
                commons = np.array(w2v.wv.most_similar(df_temp.token,topn=df_temp.shape[0]))[:,0]
                df_temp['most_common'] = commons 
            table = pd.concat([table, df_temp])
    else:
        pass
    return table.sort_values('topic') 



def reviews2topics(w2v, keys, KM=None, k=15, Time=True):
    '''
    Converts Keywords to list topics defined by categories.
    
    w2v : trained w2v vectorizor
    keys : list of keywords that match word vectors
    KM : sklearn KMeans model (else created by default)
    Time: print time bool
    
    Returns:
    pd.DataFrame of topics
    loss : float : Sum of Distances to centroids
    KM : trained KMeans model
    
    USES: terms2vec, central_terms
    '''
    t = time()
    #create the word vectors
    X, X_labels = terms2vec(w2v, keys)

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
            n_jobs=None,
            algorithm='auto',
            )
    
    KM.fit_transform(X)
        
    losses = KM.inertia_#score(X) ########TRY INERTIA
    df_topics = central_terms(X, KM=KM, n=5, keys=X_labels[:,1], w2v=w2v, k=k); #X_labels is of form [[idx,word],...]
    if Time: print(time()-t,'s')
    return df_topics, losses, KM


def ksearch2topics(w2v, keys, krange, Time=True):
    '''
    Generate lists of topic dataframes and trained KMeans models for each value for k in a gridsearch
    
    w2v : trained w2v vectorizor
    keys : list of keywords that match word vectors
    Time: print time bool
    
    Returns:
    pd.DataFrame of topics
    loss : float : Sum of Distances to centroids
    KM : trained KMeans model
    
    USES: reviews2topics, terms2vec, central_terms
    '''
    t0 = time()
    topics_dfs = []
    losses_lst = []
    models = []
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
        topics, losses, KM2 = reviews2topics(w2v, keys, KM, k=k, Time=Time)
        
        topics_dfs.append(topics) 
        losses_lst.append(losses)
        models.append(KM2)
        
    if Time: print(time()-t0,'s')
    return topics_dfs, losses_lst, models, krange



## ----------- PCA VISUALIZATION ------------###


# def table2topNwords(DF, topN):
#     '''
#     From a pd.DataFrame, get top words for each topics
    
#     df : pd.DataFrame : columns=['topic', 'distance', 'token', 'most_common']
#     topN : int : max words per topic to extract
    
#     Returns:
#     topN : list of topN*2 words for all categories
#     top2 : list of top 2 words for all categories
#     k : number of topics
#     '''
#     df = DF.copy()
#     topNs = set()
#     top2s = set()
#     for t in df.topic.unique():
#         temp = df[df.topic ==t]
#         topNs.update(list(temp.token.iloc[:topN]) + list(temp.most_common.iloc[:topN]))
#         top2s.update([temp.token.iloc[0] , temp.most_common.iloc[0]])
#     k= len(df.topic.unique())
#     return topNs, top2s, k


def generate_colors(n=20, cmap='nipy_spectral'):
    '''
    n : length of array
    map : str : matplotlib color style (suggested alternative: 'prism')
    
    Returns: list of unique colors for plotting
    '''
    cm = plt.get_cmap(cmap)
    colors = np.array([cm(1.*i/n) for i in range(n)])
    return colors


def table2topBOW(DF, n):
    '''
    Convert a pd.DataFrame to list of topN words and top2 words for each topic
    '''
    df = DF.copy()
    m= (n/2)
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
