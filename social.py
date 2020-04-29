# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:38:01 2020

@author: Sunshine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

dataset = pd.read_csv(r"C:\Users\Sunshine\Downloads\Compressed\Machine-Learning-A-Z-New\Machine Learning A-Z New\Part 3 - Classification\Section 18 - Naive Bayes\Social_Network_Ads.csv")

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X, y)


pickle.dump(classifier, open('social.pkl', 'wb'))
model = pickle.load(open('social.pkl', 'rb'))

print(model.predict([[32, 150000]]))