from collections import defaultdict
import csv
import random
import os
import email
import math
import collections
from nltk.corpus import stopwords

def load_data(path):
    with open(path) as csvfile:
        reader = list(csv.reader(csvfile))
    return reader

def split_data(data):
    testing_data = data.copy()
    testing_percentage = 0.4
    training_percentage = 1 - testing_percentage
    training_size = len(data)*training_percentage
    training_data = []
    while len(training_data) < training_size:
        training_data.append(testing_data.pop(random.randint(0, int(len(testing_data) - 1))))
    return training_data, testing_data

def test_accuracy(data):
    pass


class SpamFilter:

    def __init__(self, raw_data, smoothing):
        self.raw_data = raw_data
        self.smoothing = smoothing
        self.vocab = set()
        self.spam_words = defaultdict(int)
        self.ham_words = defaultdict(int)


    def clean_data(self):
        pass

    def train(self):
        pass

    def predict(self, text):
        pass

data = load_data('spam.csv')
split_data(data)