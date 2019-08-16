import time
import numpy as np
import pandas as pd
import pickle
import spacy
nlp = spacy.load("en_core_web_sm")
from flask import Flask, render_template
app = Flask(__name__)

#LOAD Sample Products
with open('/home/ubuntu/Opin-private/data/300sample_products_metadata.pkl', mode='rb') as f:
    sample_products = pickle.load(f)


table_style = '''<style> table, th { border: 1px solid black;} </style><style> td { border: 1px solid black; </style>'''
star_style = '''<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><style> .checked {color: orange;}</style>'''
imgs = ['https://images.homedepot-static.com/productImages/3a60b4d7-9c57-4bd7-b10b-88419c2936dd/svn/black-ilive-bluetooth-speakers-ihb23b-64_1000.jpg',
       'https://images.qvc.com/is/image/e/43/e232643.001',
       'https://media.wired.com/photos/5b8837d22997aa2df18d9970/master/pass/roomba.jpg',
       'https://images.qvc.com/is/image/e/49/e228049.001',
        'https://i02.hsncdn.com/is/image/HomeShoppingNetwork/prodfull/apple-macbook-pro-133-256gb-ssd-laptop-wtouch-bar-and-a-d-20190604173503487~9154035w_040.jpg',
        'https://static.grainger.com/rp/s/is/image/Grainger/54TW79_AW01?$mdmain$',
       'https://my-test-11.slatic.net/original/803ea4ffc66925f04fbb0854c1d0f865.jpg']

def revs2html(tok_sentDF, rev_ids, words):
    revshtml = []
    for revid in rev_ids:
        try: row = tok_sentDF[tok_sentDF.review_id==revid].iloc[0]
        except: row = tok_sentDF[tok_sentDF.review_id==revid]
        header = '<b>' + row.review_headline + '</b><body>  '+ str(row.review_date) +'</body>'
        text = '<br><br>'+ ' '.join(['<mark><b>' +tok.text+ '</b></mark>' if tok.lemma_ in words
                                     else tok.text for tok in nlp(str(row.review_body))])
        stars_html = ''.join(['<span class="fa fa-star checked"></span>' for _ in range(int(round(row.stars,0)))])
        cell = '<tr><td>'+header+'<br>'+stars_html+text+'</tr></td>'
        revshtml.append(cell)
    html = table_style+star_style+'<table>' + ''.join([text for text in revshtml]) + '</table>'
    return html


def sentiment_table(productDF, product_id): #after clicking on one product
    table_content = []
    red = 'rgba(255,0,0,0.7)'
    green = 'rgba(0,255,0,0.7)'
    product_title = str(productDF.product_title[0])
    n_reviews = productDF.review_count[0]
    stars = round(productDF.ave_stars[0],2)
    sentimentDF = productDF.scored_clusters.iloc[0]
    cluster_dict = {k:v for k,v in productDF.renamed_clusters[0].items() 
                            if k in list(sentimentDF.index)}
    
    for key, words in cluster_dict.items():
#     return ' '.join([word for word in debug])
        #neg-pos ratio
        p = sentimentDF.loc[key,'p_mentions']
        n = sentimentDF.loc[key,'n_mentions']
        np_ratio = n/(p+n)*100
        
        cell_style = '"background-image:linear-gradient(90deg, {} 0%, {} {}%, {} {}%, {} 100%);" ' \
                                        .format(red,red, np_ratio, green,np_ratio+1,green)
        key_words = ' '.join([word for word in key.split('/')])
        kw_cell = '<td style =' + cell_style + '><center>' + \
            '<button><b><a href="/filter_reviews/' + product_id + '/' + key_words + \
                '/' + ' '.join([word for word in words]) + '"> '+ \
                key_words + '</a></b></center></button></td>' #
            #method="POST" type="submit" name="user_input" action="/word_counter"
        
        row = '<tr>' + kw_cell + '</tr>'
        table_content.append(row)
        table_html = table_style + '<center><table cellpadding="4"><tbody>' + \
                ''.join([row for row in table_content]) + '</tbody></table></center>'

    html = '<h2><center>' + product_title + '<br>' + str(n_reviews) + ' reviews <br>' \
             + str(stars) + ' stars</center></h2>' + table_html
    return html

def products_table(): #to generate products homepage
    products_html = []
    for i,row in sample_products.iterrows():
        np.random.choice(imgs, 1)
        if len(row.product_title)>80:
            title = '<center><h3>' + str(row.product_title[:80]) + '...</h3></center>'
        else: title = '<center><h3>' + str(row.product_title) + '</h3></center>'
        img_src = str(np.random.choice(imgs, 1)[0])
        img = '<center><img src="' + img_src + '" alt="sample_product" width="200" height="200"></center>'
        stars = '<center>' + ''.join(['<span class="fa fa-star checked"></span>' for 
                         _ in range(int(round(row.ave_stars,0)))]) + '</center>'
        link = '/product_page/'+str(row.product_id)
        reviews = '<center><h3><a href="'+ link + '">' + str(int(row.review_count))+ ' Reviews</a></h3></center>'

        product_html = '<td>' + title + img + stars + reviews + '</td>'
        products_html.append(product_html)
    for i, item in enumerate(products_html):
        if i==0:
            products_html[i] = '<tr>'+item
        elif (i%5 == 0):
            products_html[i] = '</tr><tr>'+item
    products_html[-1] = products_html[-1]+'</tr>'
    table = table_style+star_style +'<center><table style="max-width:90%;">' \
            + ''.join([html for html in products_html]) + '</table></center>'
    return table

@app.route('/')
def root():
    background = '''<style> body  {
  background-image: url("Screen Shot 2019-08-13 at 5.58.13 PM.png");
  background-repeat: no-repeat;
  background-attachment: fixed;
    }</style>'''
    return background + '<body>'+ Electronics() + '</body>'

@app.route('/Electronics')
def Electronics():
    return products_table()
#     return render_template('index.html')

@app.route('/product_page/<product_id>') #make this product_page
def product_page(product_id):
    one_metadata = sample_products[sample_products.product_id==product_id]
    return sentiment_table(one_metadata, product_id)

   
# FILTERED REVIEWS
@app.route('/filter_reviews/<product_id>/<category>/<words_str>' ) 
def filter_reviews(product_id, category, words_str,):
    productDF = sample_products[sample_products.product_id==product_id]
    tok_sentDF = productDF.toks_sents[0]
    
#     tok_sentDF = sample_products[sample_products.product_id==product_id].toks_sents[0]
    words = [word.strip("'").strip() for word in words_str.strip('[').strip(']').split(' ')] 
    p_rev_ids = list(tok_sentDF[tok_sentDF.token.isin(words) & (tok_sentDF.sentence_polarity==1)].review_id.unique())
    n_rev_ids = list(tok_sentDF[tok_sentDF.token.isin(words) & (tok_sentDF.sentence_polarity==-1)].review_id.unique())

    p_table = revs2html(tok_sentDF, p_rev_ids, words)
    n_table = revs2html(tok_sentDF, n_rev_ids, words)
    table = '<center><table style="max-width:85%;"><tr><td bgcolor="#baffad" width="50%" valign="top">' + p_table + \
            '</td><td bgcolor="#c78585" width="50%" valign="top">' + n_table + '</td></tr></table></center>'
    html = '<center><h2>Topic</h2><b>'+category+'</b><h3>Search Terms</h3><b>'+words_str+ \
            '</b></body></center><br>' + table
    return html

if __name__ == '__main__':
    #insert while loop to handle 'OSError: [Errno 48] Address already in use'
    app.run(host='0.0.0.0', port=8080, debug=True)