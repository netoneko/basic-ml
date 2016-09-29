#!/usr/bin/env python

from sklearn.externals import joblib
model = joblib.load('model.pkl')

print model.predict([[0, 1]])
