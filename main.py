from collections import defaultdict
import string
import csv
import random
import os
import email
import math
import collections
from zipfile import error

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
    training_data = data.copy()
    testing_percentage = 0.6
    training_percentage = 1 - testing_percentage
    training_size = int(len(data)*training_percentage)
    testing_data = []
    while len(training_data) > training_size:
        x = random.randint(0, int(len(training_data) - 1 ))
        temp = training_data.pop(x)
        testing_data.append(temp)
    return training_data, testing_data

def test_accuracy(data):
    TP, TN, FP, FN = 0,0,0,0
    train, test = split_data(data)
    print('train len: ', len(train))
    print('test len: ',len(test))
    model = SpamFilter(train, smoothing=1)
    model.clean_data()
    # print('rawData: ', model.raw_data[:5])
    model.train()
    #print('Raw: \n\t', test)
    result = []
    for x in test:
        result.append(model.predict(x[1]))
    #print('\n\nResult: \n\t', result)
    for index, value in enumerate(result):
        if value[0] == 'ham' and test[index][0] == 'ham':
            TP += 1
        elif value[0] == 'ham' and test[index][0] == 'spam':
            FP += 1
        elif value[0] == 'spam' and test[index][0] == 'spam':
            TN += 1
        elif value[0] == 'spam' and test[index][0] == 'ham':
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(test[:5])
    print(result[:5])
    print('{:<10} {:>10} {:>10}'.format('','Pred Pos','Pred Neg'))
    print('{:<10} {:>10} {:>10}'.format('True Pos', TP, FN))
    print('{:<10} {:>10} {:>10}'.format('True Neg',FP,TN))

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)


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
        temp_string = ''
        for x in self.raw_data:
            temp_tokens = []
            temp_string = x[1].lower()
            for y in string.punctuation:
                temp_string = temp_string.replace(y, "")
            temp_string = temp_string.strip()
            temp_string = temp_string.split()
            for y in temp_string:
                if not y in stopwords.words('english'):
                    temp_tokens.append(y)
            for y in temp_tokens:
                self.vocab.add(y)
            text.append([x[0], temp_tokens])
        self.raw_data = text
    pass

    def train(self):
        for x in self.vocab:
            self.ham_words[x] += self.smoothing
            self.spam_words[x] += self.smoothing
        for x in self.raw_data:
            if x[0] == 'ham':
                for y in x[1]:
                    self.ham_words[y] += 1
            else:
                for y in x[1]:
                    self.spam_words[y] += 1
        pass



    def predict(self, text):   #Complete the predict method so it can receive a text string and will return a ham or spam classification. Use a logarithmic scale.
        # https://www.saedsayad.com/naive_bayesian.htm
        #P(A|B) = P(AnB)/P(B)   A given B
        #P(c|X) = P(x1|c)*P(x2|c)*...*P(xn|c)*P(c)
        #P(c|x) = P(x|c)P(c) / P(x)
        #P(c | x) posterior probability of class(target) given predictor(attribute)
        #P(x | c) prior probability of class
        # P(c) is the likelihood which is the probability of predictor given class
        # P(x) is the prior probability of predictor
        temp_string = ''
        temp_tokens = []
        temp_string = text.lower()
        total_spam = 0
        total_ham = 0
        total_words_spam = sum(self.spam_words.values())
        total_words_ham = sum(self.ham_words.values())
        p_spam = 0
        p_ham = 0
        for y in string.punctuation:
            temp_string = temp_string.replace(y, "")
        temp_string = temp_string.strip()
        temp_string = temp_string.split()
        for y in temp_string:
            if not (y in stopwords.words('english')) and (y in self.vocab):
                temp_tokens.append(y)
        for x in self.raw_data:
            if x[0] == 'ham':
                total_ham += 1
            elif x[0] == 'spam':
                total_spam += 1
        p_spam = math.log(total_spam/(total_ham+total_spam))
        p_ham = math.log(total_ham/(total_spam+total_ham))
        #P(A|B) = P(AnB)/P(B)   A given B
        p_word_spam = []
        p_word_ham = []
        for x in temp_string:
            if x in self.vocab:
                #print(total_words_spam, self.spam_words[x], self.ham_words[x])
                p_word_spam.append([x, math.log(self.spam_words[x] / total_words_spam)])
                p_word_ham.append([x, math.log(self.ham_words[x] / total_words_ham)])

        #print('HAM', p_word_ham)
        #print('T HAM', total_words_ham)
        #print('SPAM', p_word_spam)
        #print('T SPAM', total_words_spam)
        sum_p_word_spam = p_spam
        sum_p_word_ham = p_ham
        for x in p_word_spam:
            sum_p_word_spam += x[1]
        for x in p_word_ham:
            sum_p_word_ham += x[1]
        #print(f"spam: {sum_p_word_spam} vs ham: {sum_p_word_ham}")
        if sum_p_word_ham > sum_p_word_spam:
            return ['ham', text]
        else:
            return ['spam', text]

data = load_data('spam.csv')[1:]
test_accuracy(data)
test_accuracy(data)
test_accuracy(data)
test_accuracy(data)
