# %%
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import sys
import math
from nltk.stem import PorterStemmer
from sklearn.datasets import load_iris
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
import numpy as np
import tensorflow as tf
import sys

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

ps = PorterStemmer()

# %%
def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word

# %%
def preprocess_sentences(sentence):
    processed_sentence = []
    sentence = sentence.lower()
    regExp = re.compile(r'[.,;_()"/\']',re.DOTALL)
    sentence = regExp.sub("",tweet)
    sentence = re.sub(r'[^\x00-\x7F]+','', tweet)
    words = sentence.split()
    for word in words:
        word = preprocess_word(word)
        if word not in stop_words:
            word = str(ps.stem(word))
        processed_sentence.append(word)

    return ' '.join(processed_sentence)

# %%
def prepare_data(data):
    data=data.strip(',\n');
    #print(data)
    words=data.split()
    #print(words[1])
    question=""
    for i in range(len(words)-1):
        if i==0:
            question+=words[i]
        else:
            question+=" "+words[i]
    #print(question)
    label=words[-1];
    return question,label

# %%
processed_file=open("processed_questions.csv","w");

dict={'Sentence':[],'Validity':[]}
processed_file.write("Sentence"+" "+'Validity'+'\n');
for i in range(len(df['sent'])):
    question=preprocess_tweets(df['sent'][i]);
    if len(question)<=1: continue;
    dict['Sentence'].append(question);
    dict['Validity'].append(df['label'][i])

fp.close()
processed_file.close()        

# %%
data=pd.DataFrame(dict)

# %%
data.to_csv("process_questions.csv",index=False,header=True)

# %%
df


# %%


# %%
stop_words = set(stopwords.words('english'))

# %%
df=pd.read_csv("process_questions.csv")
#df=df.drop(['Unnamed: 2'], axis = 1) 

# %%
X_train, X_test, y_train, y_test = train_test_split(df['Sentence'],df['Validity'], test_size=0.2,random_state=33)

# %%
corpus=df['Sentence']
uni_grams=[]
for sentence in corpus:
    words=sentence.split()
    for word in words:
        if word not in stop_words:
            uni_grams.append(ps.stem(word))
vocabsize=len(uni_grams)

# %%


# %%
vocab={}
for word in uni_grams:
    if word not in vocab:
        vocab[word]=0;
    vocab[word]+=1


# %%
vocab_size=len(vocab)
dim=200

# %%
def get_feature_vector(sentence):
    words = sentence.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector

# %%
def get_features(data):
    global uni_grams,vocabsize
    new_data=[]
    for sentence in data:
        words=sentence.split()
        features=[]
        for word in words:
            if word not in stop_words:
                features.append(ps.stem(word))
        new_data.append(features)
    trainsize=len(data)
    features = lil_matrix((trainsize, vocabsize))
    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            features[i,j]+=1;
    return features

# %%
def get_glove_vectors(vocab):
    print('Looking for GLOVE vectors')
    glove_vectors = {}
    found = 0
    with open('glove.twitter.27B.200d.txt', 'r') as glove_file:
        for i, line in enumerate(glove_file):
            #utils.write_status(i + 1, 0)
            tokens = line.split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print('\n')
    print('Found %d words in GLOVE',found)
    return glove_vectors

# %%
glove_vectors = get_glove_vectors(vocab)

# %%
 embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01

# %%
for word, i in vocab.items():
        glove_vector = glove_vectors.get(word)
        if glove_vector is not None:
            embedding_matrix[i] = glove_vector

# %%
X_train

# %%
X_train=np.array(X_train)
y_train=np.array(y_train)

# %%
X_test=np.array(X_test)
y_test=np.array(y_test)

# %%
sentences=[]
for sentence in X_train:
    sentences.append(get_feature_vector(sentence))

# %%
labels=[]
for label in y_train:
    labels.append(label)
labels=np.array(labels)

# %%
sentences = pad_sequences(sentences, maxlen=40,padding='post')

# %%
shuffled_indices = np.random.permutation(sentences.shape[0])

# %%
sentences = sentences[shuffled_indices]
labels = labels[shuffled_indices]

# %%
# LSTM model To classify a sentence is questionable or not with glove pre-trained W2V embedding 

def Classication_Model():
    model = Sequential()
    model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=40))
    model.add(Dropout(0.25))
    model.add(LSTM(128))
    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

# %%
model = Classication_Model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
history = model.fit(sentences, labels,validation_split = 0.1, epochs=50, batch_size=4)

# %%
"""
# Plots
"""
# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# %%

x1 = np.array(epoches)
y1 = np.array(val_acc)
x1_smooth = np.linspace(x1.min(), x1.max(), 500)
y1_smooth = spline(x1, y1, x1_smooth)

# %%
for h in history.history:
    print(h)

# %%
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.000001)
                                  
model.fit(sentences, labels, batch_size=200,epochs=20,shuffle=True,callbacks=[reduce_lr])

# %%
test_sentences=[]
for sentence in X_test:
    test_sentences.append(get_feature_vector(sentence))

# %%
ground_truth=[]
for label in y_test:
    ground_truth.append(label)
ground_truth=np.array(ground_truth)

# %%
test_sentences = pad_sequences(test_sentences, maxlen=40,padding='post')

# %%
prediction = model.predict(test_sentences, batch_size=128, verbose=1)

# %%


# %%


# %%
results = zip(map(str, range(len(test_sentences))), np.round(prediction[:, 0]).astype(int))

# %%
output_labels=np.round(prediction[:, 0]).astype(int)

# %%
correct = 0
correct += np.sum(output_labels == ground_truth)
Accuracy=correct/len(prediction)
print(Accuracy)
# %%


# %%


# %%

# %%


# %%
################# Based on TF-IDF representation of sentence ########################
# %%

X_train=get_features(X_train)

# %%
transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
transformer.fit(X_train)

# %%
X_train.todense()

# %%
X_train = transformer.transform(X_train)

# %%
X_train.todense()

# %%
X_test=get_features(X_test)

# %%
transformer.fit(X_test)
X_test = transformer.transform(X_test)
y=np.array(y_train)

# %%
"""
# Naive Bayes
"""

# %%
# train data
clf = MultinomialNB()
clf.fit(X_train,y)
y_test=np.array(y_test)

# test data
correct = 0
prediction = clf.predict(X_test)
correct += np.sum(prediction == y_test)
Accuracy=correct/len(prediction)
print(Accuracy)
# %%
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
# %%
train_X = df['Sentence']
train_y = df['Validity']
train_X=get_features(train_X)
train_y = np.array(train_y)

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

title = "Learning Curves (Multinomial Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
plot_learning_curve(clf, title, X_train, train_y, axes=axes[:], ylim=(0.7, 1.0),cv=cv, n_jobs=2)

# %%
"""

