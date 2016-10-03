#!/usr/bin/env python

from sklearn.externals import joblib
import sys

CHARACTERS = joblib.load('characters.pkl')
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

result = model.predict(vectorizer.transform([sys.argv[1]]))

print CHARACTERS[result[0]]
