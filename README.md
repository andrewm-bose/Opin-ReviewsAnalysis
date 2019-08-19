[![python3.7.3](https://img.shields.io/badge/python-3.7.3-orange)](https://spacy.io)
[![spaCy](https://img.shields.io/badge/-spaCy-blue)](https://spacy.io)
[![gensim](https://img.shields.io/badge/gensim-Word2Vec-blue)](https://radimrehurek.com/gensim/)
[![nltk](https://img.shields.io/badge/-nltk-orange)](https://www.nltk.org)


[![vaderSentiment](https://img.shields.io/badge/-vaderSentiment-24292E)](https://github.com/cjhutto/vaderSentiment)


[![flask](https://img.shields.io/badge/-flask-363B3D)](https://palletsprojects.com/p/flask/)



# Opin - Amazon reviews filtering by Topics & Sentiment

### This is an attempt at replicating the amazon reviews key-word filtering buttons, using sentiment analysis and topic modeling techniques with word embeddings.

<INSERT EXAMPLE AMAZON REVIEW BUTTONS>
![Example Amazon review buttons](https://github.com/andrewm-bose/Opin-ReviewsAnalysis/blob/master/flaskapp/imgs/Bose_Home_Speaker_500_Black_1.jpg)

### Main libraries used - pandas, spacy, gensim, nltk, matplotlib, seaborn

## This project to still under construction. However, the code runs. Feel free to explore the jupyter notebooks:
> Preprocess2py.ipynb - cleans and tokenizes amazon electronics reviews dataset - 3.2M reviews, 62k products
> Vectorize2py.ipynb - uses gensim word2vec to create word embeddings from preprocessed reviews
> Main2py.ipynb - main program, performs topic modelling and sentiment analysis on a random product from the data set. Output ~15 topics with the customers' sentiment counts for the key words in that cluster.
> Main2y-multirun.ipynb - same as Main2py but runs on 1000 randomly select products.
> flaskapp.opin_app.py - runs a basic website to demonstrate the functionality of the model.

## This readme is also still under construction. Everything will be more thoroughly explained in the coming weeks.

### Thanks for visiting!
