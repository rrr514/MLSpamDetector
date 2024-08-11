import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
# nltk.download('stopwords')

import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

from model_training import remove_punctuations, remove_stopwords, max_len

# Loading the model
model = tf.keras.models.load_model('my_model.keras')

# Reading the test data
data = pd.read_csv('spam_ham_testset.csv')

print(data.head())

texts_to_classify = data['text']

# Preprocessing the data
# texts_to_classify = texts_to_classify.replace("\n", " ").replace("\t", " ")
texts_to_classify = texts_to_classify.apply(lambda x: remove_punctuations(x))
texts_to_classify = texts_to_classify.apply(lambda x: remove_stopwords(x))

print(texts_to_classify.head())

# Convert text to sequences using tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_to_classify)

sequences_to_classify = tokenizer.texts_to_sequences(texts_to_classify)
sequences_to_classify = pad_sequences(sequences_to_classify, maxlen=max_len, padding='post', truncating='post')

# Make predictions
predictions = model.predict(sequences_to_classify)
predicted_labels = ['spam' if x > 0.5 else 'ham' for x in predictions]

print(predicted_labels)