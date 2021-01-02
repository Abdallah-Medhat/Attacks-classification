# -*- coding: utf-8 -*-
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('payloads.csv')

# Encoding categorical data
# 3-grams CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def get3Grams(dataset_obj):
    payload = str(dataset_obj)
    ngrams = []
    for i in range(0,len(payload)-3):
        ngrams.append(payload[i:i+3])
    return ngrams
tfidf_vectorizer_3grams = TfidfVectorizer(tokenizer=get3Grams)
X = tfidf_vectorizer_3grams.fit_transform((dataset['payload']).values.astype('U'))

#Encoding the Dependent Variable
Y =dataset['injection_type']
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y= LabelEncoder()
Y = labelencoder_Y.fit_transform(Y).values.astype('U')

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# # Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# # Predicting the Test set results
y_pred_train = classifier.predict(X_train)
y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(y_train, y_pred_train)
cm = confusion_matrix(y_test, y_pred)

# #accuracy
accuracy_tain = (cm_train[0][0] + cm_train[1][1]+cm_train[2][2]+cm_train[3][3]) / (cm_train.sum().sum())
accuracy = (cm[0][0] + cm[1][1]+cm[2][2]+cm[3][3]) / (cm.sum().sum())



