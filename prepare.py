#!/usr/bin/env python

import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from IPython import embed

import sqlite3
conn = sqlite3.connect('./scripts/db/seinfeld.sqlite')
cursor = conn.cursor()

CHARACTERS = {
    'JERRY': 0,
    'GEORGE': 1,
    'ELAINE': 2,
    'KRAMER': 3
}

def get_lines(character, limit=100, offset=0, length=25):
    cursor.execute('SELECT text FROM utterance \
    WHERE speaker=? \
    AND length(text) > ? \
    ORDER BY RANDOM() DESC \
    LIMIT ? OFFSET ?', [character, length, limit, offset])

    return map(lambda l: l[0].lower(), cursor.fetchall())

vectorizer = TfidfVectorizer(min_df=1,
    analyzer='word',
    stop_words='english',
    lowercase='true')

TRAIN_LIMIT = 15000

jerry_lines = get_lines('JERRY', limit=TRAIN_LIMIT)
george_lines = get_lines('GEORGE', limit=TRAIN_LIMIT)
elaine_lines = get_lines('ELAINE', limit=TRAIN_LIMIT, length=10)
kramer_lines = get_lines('KRAMER', limit=TRAIN_LIMIT)

lines = jerry_lines + george_lines + elaine_lines + kramer_lines

features = vectorizer.fit_transform(lines)
answers = np.array([CHARACTERS['JERRY']] * len(jerry_lines)
    + [CHARACTERS['GEORGE']] * len(george_lines)
    + [CHARACTERS['ELAINE']] * len(elaine_lines)
    + [CHARACTERS['KRAMER']] * len(kramer_lines))

model = SGDClassifier()
model.fit_transform(features, answers)
model.densify()

print model

print 'accuracy on same set', accuracy_score(answers, model.predict(features))

def test(character, test_limit=500):
    test_lines = get_lines(character, limit=test_limit)

    test_features = vectorizer.transform(test_lines).toarray()
    test_answers = np.array([CHARACTERS[character]] * len(test_lines))
    test_predictions = model.predict(test_features)

    return accuracy_score(test_answers, test_predictions)

print 'accuracy on test set (jerry)', test('JERRY')
print 'accuracy on test set (george)', test('GEORGE')
print 'accuracy on test set (elaine)', test('ELAINE')
print 'accuracy on test set (kramer)', test('KRAMER')

# embed()

joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')
joblib.dump({v: k for k, v in CHARACTERS.items()}, 'characters.pkl')
