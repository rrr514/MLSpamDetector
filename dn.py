import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
# from wordcloud import wordcloud
# nltk.download('stopwords')

import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

# loading in the data set
data = pd.read_csv('spam_ham_dataset.csv')

# information about the data set
print(data.head())
print(data.shape)
# sns.countplot(x='label', data=data)
# plt.show()

# Preprocessing the data

# removing the 'Subject' word from the text
data['text'] = data['text'].str.replace('Subject', '')
print(data.head())

# removing the punctuations from the text
punctuations_list = string.punctuation

def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

data['text'] = data['text'].apply(lambda x: remove_punctuations(x))
print(data.head())

# removing stopwords from the text
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    
    imp_words = []

    # store the important words
    for words in str(text).split():
        if words.lower() not in stop_words:
            imp_words.append(words)

        output = " ".join(imp_words)

        return output
    
data['text'] = data['text'].apply(lambda x: remove_stopwords(x))
print(data.head())