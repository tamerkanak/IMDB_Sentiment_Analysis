# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 01:08:01 2023

@author: tamer
"""
#1. library
import pandas as pd

#2. data loading
data = pd.read_csv("IMDBDataset.csv")

#3. preprocessing
x = data.iloc[:,0]  
y = data.iloc[:,1]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#4.NLTK to root the word and get stopwords
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stop = nltk.download("stopwords")

from nltk.corpus import stopwords
import re

#5. rooting of words
textList = []
for i in range(50000):
    text = re.sub("[^a-zA-Z]", " ", x[i])
    text = text.lower()
    split = text.split()
    text = [ps.stem(i) for i in split if not i in set(stopwords.words("english"))]
    text = " ".join(text)
    textList.append(text)
    
#6. converting sentences to numeric values
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(textList).toarray()

#7. splitting data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(vectorized,y,test_size=0.33,random_state=0)

#8. XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)

#9. Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)
print(cm)







