#!/usr/bin/env python

import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

X = np.array([[0, 1, 1], [1, 0, 0]])
model = LogisticRegression()

model.fit(X[:,np.array([True, True, False])], X[:,-1])

joblib.dump(model, 'model.pkl')
