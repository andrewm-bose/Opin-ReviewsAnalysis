# Opin - Amazon reviews filtering by Topics & Sentiment

## This is an attempt at replicating the amazon reviews key-word filtering buttons, using sentiment analysis and topic modeling techniques.

(https://huggingface.co/coref/assets/thumbnail-large.png)](https://huggingface.co/coref/)

### Main libraries used - pandas, spacy, gensim, nltk, matplotlib, seaborn

## This project to still under construction. However, the code runs. Feel free to explore the jupyter notebooks:
> Preprocess2py.ipynb - cleans and tokenizes amazon electronics reviews dataset - 3.2M reviews, 62k products
> Vectorize2py.ipynb - uses gensim word2vec to create word embeddings from preprocessed reviews
> Main2py.ipynb - main program, performs topic modelling and sentiment analysis on a random product from the data set. Output ~15 topics with the customers' sentiment counts for the key words in that cluster.
> Main2y-multirun.ipynb - same as Main2py but runs on 1000 randomly select products.
> flaskapp.opin_app.py - runs a basic website to demonstrate the functionality of the model.

## This readme is also still under construction. Everything will be more thoroughly explained in the coming weeks.

### Thanks for visiting!
