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

# loading in the data set
data = pd.read_csv('spam_ham_dataset.csv')

# information about the data set
print(data.head())
print(data.shape)
# sns.countplot(x='label', data=data)
# plt.show()

#Balancing the data set
ham_msg = data[data.label == 'ham']
spam_msg = data[data.label == 'spam']
ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)

balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)
# plt.figure(figsize=(8, 6))
# sns.countplot(data = balanced_data, x = 'label')
# plt.show()

# Preprocessing the data

# removing the 'Subject' word from the text
data['text'] = data['text'].str.replace('Subject', '')
print('Text after removing Subject: ', data['text'].iloc[0])

# removing the punctuations from the text
punctuations_list = string.punctuation

def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

data['text'] = data['text'].apply(lambda x: remove_punctuations(x))
print('Text after removing punctuation: ', data['text'].iloc[0])

# removing stopwords from the text
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    
    imp_words = []

    # store the important words
    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)

    return output
    
data['text'] = data['text'].apply(lambda x: remove_stopwords(x))
print('Text after removing stopwords:', data['text'].iloc[0])

# def plot_word_cloud(data, typ):
#     email_corpus = " ".join([text for text in data['text'] if text is not None])

#     plt.figure(figsize=(7, 7))

#     wc = WordCloud(background_color = 'black',
#                    max_words = 100,
#                    width = 800,
#                    height = 400,
#                    collocations=False).generate(email_corpus)
    
#     plt.imshow(wc, interpolation='bilinear')
#     plt.title(f'WordCloud for {typ} emails', fontsize=15)
#     plt.axis('off')
#     plt.show()

# plot_word_cloud(data[data['label_num'] == 0], typ='ham')
# plot_word_cloud(data[data['label_num'] == 1], typ='spam')

# Split the data into training and testing data
train_X, test_X, train_Y, test_Y = train_test_split(data['text'], data['label_num'], test_size=0.2, random_state=42)

# Remove None types and empty strings from train_X and test_X
train_X_filtered = [text for text in train_X if text and text.strip() != '']
test_X_filtered = [text for text in test_X if text and text.strip() != '']

# Filter corresponding labels to ensure consistency
train_Y_filtered = [label for text, label in zip(train_X, train_Y) if text and text.strip() != '']
test_Y_filtered = [label for text, label in zip(test_X, test_Y) if text and text.strip() != '']

# Reassigning the filtered data
train_X, train_Y = train_X_filtered, train_Y_filtered
test_X, test_Y = test_X_filtered, test_Y_filtered

train_Y = np.array(train_Y)
test_Y = np.array(test_Y)

print(f"Shape of train_Y: {train_Y.shape}")
print(f"Shape of test_Y: {test_Y.shape}")

print(train_X[:5])

# Tokenizing the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

print(f"Vocabulary size: {len(tokenizer.word_index)}")

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)

# Pad sequences to have the same length
max_len = 100
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

print(f"Shape of train_sequences: {train_sequences.shape}")
print(f"Shape of test_sequences: {test_sequences.shape}")

# Building the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.build(input_shape=(None, max_len))   

# Print the model summary
model.summary()

# Compile the model
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = False),
              metrics = ['accuracy'],
              optimizer = 'adam')

# Callbacks
es = EarlyStopping(patience=3, monitor = 'val_accuracy',
                   restore_best_weights=True)

lr = ReduceLROnPlateau(patience=2, monitor = 'val_loss',
                       factor = 0.5, verbose = 0)

# Training the model
history = model.fit(train_sequences, train_Y, validation_data=(test_sequences, test_Y),
                    epochs=20, batch_size=32, callbacks=[es, lr])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss: ', test_loss)
print('Test Accuracy: ', test_accuracy)

# Plotting the model accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
