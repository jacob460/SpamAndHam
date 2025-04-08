from collections import defaultdict
import string
import csv
import random
import os
import email
import math
import collections
from nltk.corpus import stopwords

def load_data(path):
    try:
        with open(path) as csvfile:
            reader = list(csv.reader(csvfile))
    except UnicodeDecodeError:
        with open(path, encoding='latin-1') as csvfile:
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
        #print(string.punctuation)
        text = []
        for x in self.raw_data:
            if not (x in stopwords.words('english')):
                temp_string = x[1].lower()
                for y in string.punctuation:
                    temp_string = temp_string.replace(y, "")
                temp_string = temp_string.strip()
                temp_string = temp_string.strip()
                temp_string = temp_string.split()
                for y in temp_string:
                    self.vocab.add(y)
            text.append([x[0], temp_string])
        self.raw_data = text
    pass

    def train(self):
        for x in self.raw_data:
            if x[0] == 'ham':
                for y in x[1]:
                    self.ham_words[y] += 1
            else:
                for y in x[1]:
                    self.spam_words[y] += 1


    def predict(self, text):
        pass

data = load_data('spam.csv')[1:]
train, test = split_data(data)
model = SpamFilter(data, smoothing=1)
model.clean_data()
print('rawData: ', model.raw_data[:5])
model.train()
print(f'HAM: \n{model.ham_words}\n\nSPAM: \n{model.spam_words}')
