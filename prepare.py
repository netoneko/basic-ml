#!/usr/bin/env python

import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from IPython import embed
import sys

import sqlite3
conn = sqlite3.connect('./scripts/db/seinfeld.sqlite')
cursor = conn.cursor()

CHARACTERS = {
    'JERRY': 0,
    'GEORGE': 1,
    'ELAINE': 2,
    'KRAMER': 3
}

def get_lines(character, limit=100, offset=0, words=6):
    cursor.execute('SELECT text FROM utterance \
    WHERE speaker=? \
    ORDER BY RANDOM() DESC \
    LIMIT ? OFFSET ?', [character, limit, offset])

    return filter(lambda l: len(l.split(' ')) >= words,
        map(lambda l: l[0].lower(), cursor.fetchall()))

vectorizer = TfidfVectorizer(min_df=1,
    analyzer='word',
    stop_words='english',
    lowercase=True,
    binary=False)

TRAIN_LIMIT = 16000

def get_features():
    jerry_lines = get_lines('JERRY', limit=TRAIN_LIMIT, words=4)
    george_lines = get_lines('GEORGE', limit=TRAIN_LIMIT, words=3)
    elaine_lines = get_lines('ELAINE', limit=TRAIN_LIMIT, words=0)
    kramer_lines = get_lines('KRAMER', limit=TRAIN_LIMIT, words=0)

    lines = jerry_lines + george_lines + elaine_lines + kramer_lines

    features = vectorizer.fit_transform(lines)
    answers = np.array([CHARACTERS['JERRY']] * len(jerry_lines)
        + [CHARACTERS['GEORGE']] * len(george_lines)
        + [CHARACTERS['ELAINE']] * len(elaine_lines)
        + [CHARACTERS['KRAMER']] * len(kramer_lines))

    return features, answers

def build_model(features, answers):
    model = SGDClassifier(n_iter=15,
        penalty='l2',
        epsilon=0.2,
        alpha=0.00001,
        loss='hinge')
    model.fit(features, answers)
    model.densify()

    return model

def test_character(character, test_limit=500):
    test_lines = get_lines(character, limit=test_limit)

    test_features = vectorizer.transform(test_lines).toarray()
    test_answers = np.array([CHARACTERS[character]] * len(test_lines))
    test_predictions = model.predict(test_features)

    return accuracy_score(test_answers, test_predictions)

def test_model(model):
    print 'accuracy on same set', accuracy_score(answers, model.predict(features))


    test_jerry = test_character('JERRY')
    test_george = test_character('GEORGE')
    test_elaine = test_character('ELAINE')
    test_kramer = test_character('KRAMER')

    # embed()

    return [test_jerry, test_george, test_elaine, test_kramer]

while(True):
    features, answers = get_features()
    model = build_model(features, answers)
    results = test_model(model)

    if all(map(lambda metric: metric > 0.64, results)):
        test_jerry, test_george, test_elaine, test_kramer = results
        print 'accuracy on test set (jerry)', test_jerry
        print 'accuracy on test set (george)', test_george
        print 'accuracy on test set (elaine)', test_elaine
        print 'accuracy on test set (kramer)', test_kramer

        joblib.dump(vectorizer, 'vectorizer.pkl')
        joblib.dump(model, 'model.pkl')
        joblib.dump({v: k for k, v in CHARACTERS.items()}, 'characters.pkl')

        sys.exit(0)
