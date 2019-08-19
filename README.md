[![python3.7.3](https://img.shields.io/badge/python-3.7.3-orange)](https://spacy.io)
[![spaCy](https://img.shields.io/badge/-spaCy-blue)](https://spacy.io)
[![gensim](https://img.shields.io/badge/gensim-Word2Vec-blue)](https://radimrehurek.com/gensim/)
[![nltk](https://img.shields.io/badge/-nltk-orange)](https://www.nltk.org)
[![vaderSentiment](https://img.shields.io/badge/-vaderSentiment-24292E)](https://github.com/cjhutto/vaderSentiment)
[![flask](https://img.shields.io/badge/-flask-363B3D)](https://palletsprojects.com/p/flask/)

# Opin - Amazon reviews filtering by Topics & Sentiment

I attempted at replicating the amazon reviews key-word filtering buttons, using sentiment analysis and topic modeling techniques with word embeddings.

#### This is a demo project and not a fulling functioning package.

<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%205.11.13%20PM.png">
> Amazon's current filter buttons

## The Goal

You can already filter Amazon customer reviews for most products by key words generated by Amazon's NLP/ML filtering algorithm. However, this is still a work in progress. It's not always easy to get the right key words, or even meaningful words.

<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%204.13.40%20PM%20copy.png">
> Not always perfect. Getting computers to understand what matter in human language is a real challenge.
  
And it looks like the buttons only extract THAT key word from the reviews, but what about misspellings and similar concepts. AND it doesn't tell you right away if these are positive or negative reviews, which I think would be valuable information when deciding which reviews I want to filter.

My model is an attempt at addressing these issues. My model generate key word filtering buttons, but each represents a topic of similar words found in the reviews. It also tells you what proportion of comments for that topic are positive or negative. It also tries prioritize topics that have a health balance of importance metric - freqently mentioned, disproportionately postive/negative reviews, and topics where there are very mixed opinions.


## To try it out, you can run either the jupyter notebook or the ipython file for the following steps. The notebooks have extra EDA with some very interesting insights:
> preprocess.ipynb/py - cleans and tokenizes amazon electronics reviews dataset - 3.2M reviews, 62k products
> vectorize.ipynb/py - uses gensim word2vec to create word embeddings from preprocessed reviews
> main_single_run.ipynb/py - main program, performs topic modelling and sentiment analysis on a random product from the data set. Output ~15 topics with the customers' sentiment counts for the key words in that cluster.
> flaskapp/opin_app.py - runs a basic demo website to demonstrate the functionality of the model.


## Examples

I threw together a basic flask app to demonstrate the functionality. It's a little rough around the edges but you can see potential of the concept.

<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%205.47.16%20PM.png">

<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%205.47.43%20PM.png">

<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%205.48.02%20PM.png">

### Behind the Scenes
<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%204.11.18%20PM.png">

<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%205.09.13%20PM.png">

## Performace

This is overall a very unsupervised machine learning task, however I developed some metrics to generally gauge how the model performs overal:
* **star rating prediction** - using the proportion of positive to negative mentions of all keywords in the generated topics, the model seems to do a pretty good job at estimating the actual star rating of the product.
* **review coverage** - defined as the proporition of reviews for a given product you'll see *at least once* by clicking through the topic buttons.

<img align="center" src="https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/readme_imgs/Screen%20Shot%202019-08-19%20at%204.10.21%20PM.png">


## Methodology
An initial dataset of <a href="https://s3.amazonaws.com/amazon-reviews-pds/readme.html"> 3.2M Amazon Electronics category customer reviews</a> is tokenized using nltk and gensim: 
>tokenizing is done with gensim.simple_preprocess, with all words lemmatized, removing stopwords, and finally using gensim Phrases to generate bigrams and trigrams.

From there, word embeddings were generated by training gensim Word2Vec on the full corpus of tokenized reviews.

From there, any random product can be extracted from the corpus, and the following techniques are applied:
>* Break the reviews into sentences using spacy, storing as a pandas DataFrame.
>* Use vader Sentiment to get the sentiment of each sentence, and use the compound score with a weighted average of the star rating to estimate the sentiment of the sentence.
>* For each token in the corpus of reviews for the one product, aggregate the positive and negative sentiment for all sentences containing that token.
>* Get the most popular keywords (tokens).
>* Using those keywords as topic (cluster) seeds, use Word2Vec to get the 6 most similar words in the corpus for that word.
>* Get the total sentiment counts for each word in each topic.
>* Take the top 15 topics based on a custom selection criteria of importance.

At the end of all that, you have your filter buttons, now filtering not but just a keyword but by topics :tada:

## Other methods
Other methods were considered, including LDA and Kmeans cluster of words into topics, but this model ended up being simpler, faster, and more consistent.

## Future improvements:
* **product level topic modeling** - can you find competitor products with better sentiment about the topic you care about?
* **product description fact extraction** - can you prioritize topics that talk about product details?
* **coreference resolution** (eg. ‘it’ died -> ‘battery’ died)
** The corpus has 3,081,927 "it"/"it's"/"its" 's! That's a lot of potential context that could given much more thorough results!
* **webscraping mechanism** to apply it to live amazon reviews/other online marketplaces
* **sub-word semantic meaning** (maybe with BERT)
* **robustly-trained sentiment classifier** (maybe with BERT)
* **gensim/spaCy Pipelines** - I also ran into memory issues with large pandas DataFrames (10M+ rows) when I exploded the reviews for all products into sentences. Both spaCy and gensim have pipeline mechanisms that let you process a corpus without holding the whole thing in memory. I'd like to impliment this to handle larger data sets, and also make the code that runs on an individual product run faster.


### Thanks for visiting!

