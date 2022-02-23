#Import useful libraries
from flask import Flask,render_template,url_for,request
import pickle
import joblib

import pandas as pd
import numpy as np
import re
from ekphrasis.classes.segmenter import Segmenter
import preprocessor as p
from time import time

import json
from collections import Counter
from wordcloud import WordCloud

import googletrans
from googletrans import Translator
from langdetect import detect, detect_langs
from langdetect import DetectorFactory
DetectorFactory.seed = 0


import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint

#Clustering
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Plotting tools
import pyLDAvis
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST', 'GET'])

def predict():
    
    def translate_words(data):
            translator = Translator()
            translator.raise_Exception = True
            data = data.apply(translator.translate, dest='en') \
                        .apply(getattr, args=('text',))
            return data
    def sent_to_words(sentences):
           for sentence in sentences:
               # deacc=True removes punctuations
               yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) 
                 if word not in stop_words] for doc in texts]
        
    # Load Saved Model
    #joblib.dump(lda_model, 'model.pkl')
    model = open('lda_model.pkl','rb')
    lda_model = joblib.load(model)
    
    translator = Translator()
    ps = PorterStemmer()
    lm = WordNetLemmatizer()
    
    stop_words = stopwords.words('english')
    
    if request.method == 'POST':
        message = request.form['message']
        tweet = [message]
    
        tweet_df = pd.DataFrame({'TweetBody':tweet}) 
        TweetBody_df  = pd.DataFrame(tweet_df['TweetBody'])
           
        TweetBody_df['TweetBody'] = translate_words(TweetBody_df['TweetBody'])

        additional  = ['RT','rt','rts','retweet', 'to','of']
        swords = set().union(stopwords.words('english'),additional)


        TweetBody_df['TweetBody_clean'] = TweetBody_df['TweetBody'].str.lower()\
                  .str.replace('(@[a-z0-9]+)\w+',' ')\
                  .str.replace('(http\S+)', ' ')\
                  .str.replace('([^0-9a-z \t])',' ')\
                  .str.replace(' +',' ')\
                  .str.replace('\d+', ' ')\
                  .apply(lambda x: [i for i in x.split() if not i in swords])

        
        TweetBody_df['TweetBody_stemmed'] = TweetBody_df['TweetBody_clean'].apply(lambda x: [ps.stem(i) for i in x if i != ''])
        
        TweetBody_df['TweetBody_lm'] = TweetBody_df['TweetBody_clean'].apply(lambda x: [lm.lemmatize(i) for i in x if i != ''])

        # segmenter using the word statistics from Twitter
        seg_tw = Segmenter(corpus="twitter")
        a = []

        for i in range(len(TweetBody_df)):
            if TweetBody_df['TweetBody_lm'][i] != a:
                listToStr1 = ' '.join([str(elem) for elem in \
                                       TweetBody_df['TweetBody_lm'][i]])
                TweetBody_df.loc[i,'TweetBody_Seg'] = seg_tw.segment(listToStr1)
       
        #Create corpus
        data = TweetBody_df["TweetBody_Seg"].values.tolist()
        data_words = list(sent_to_words(data))

        # remove stop words
        data_words = remove_stopwords(data_words)

        # Create Dictionary
        id2word = corpora.Dictionary(data_words)
        # Create Corpus
        texts = data_words
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        my_prediction = lda_model.print_topics()
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)