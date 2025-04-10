from collections import defaultdict
import string
import csv
import random
import math
import pickle
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

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
    model = SpamFilter(train, smoothing=1)
    model.clean_data()
    model.train()
    result = []
    for x in test:
        result.append(model.predict(x[1]))
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
    print('{:<10} {:>10} {:>10}'.format('','Pred Pos','Pred Neg'))
    print('{:<10} {:>10} {:>10}'.format('True Pos', TP, FN))
    print('{:<10} {:>10} {:>10}'.format('True Neg',FP,TN))

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)


    pass

def load_model():
    return pickle.load(open('model.pickle', 'rb'))


class SpamFilter:

    def __init__(self, raw_data, smoothing):
        self.raw_data = raw_data
        self.smoothing = smoothing
        self.vocab = set()
        self.spam_words = defaultdict(int)
        self.ham_words = defaultdict(int)


    def clean_data(self):
        text = []
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

    def save(self):
        with open('model.pickle', 'wb') as output:
            pickle.dump(self, output)
        pass



    def predict(self, text):
        temp_tokens = []
        temp_string = text.lower()
        total_spam = 0
        total_ham = 0
        total_words_spam = sum(self.spam_words.values())
        total_words_ham = sum(self.ham_words.values())
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
        p_word_spam = []
        p_word_ham = []
        for x in temp_string:
            if x in self.vocab:
                p_word_spam.append([x, math.log(self.spam_words[x] / total_words_spam)])
                p_word_ham.append([x, math.log(self.ham_words[x] / total_words_ham)])

        sum_p_word_spam = p_spam
        sum_p_word_ham = p_ham
        for x in p_word_spam:
            sum_p_word_spam += x[1]
        for x in p_word_ham:
            sum_p_word_ham += x[1]
        if sum_p_word_ham > sum_p_word_spam:
            return ['ham', text]
        else:
            return ['spam', text]

data = load_data('spam.csv')[1:]
test_accuracy(data)
test_accuracy(data)
test_accuracy(data)
test_accuracy(data)


## Test save and load of a model
#train, test = split_data(data)
#model = SpamFilter(train, smoothing=1)
#model.clean_data()
#model.train()

#print('m1', model.vocab)
#model.save()

#model2 = load_model()
#print('m2', model2.vocab)




# Provide your info below
student = "Jacob Semerod" # First and last name
login = "jrs460" # Your PSU login such as jdc308

# Question 1
# Approximately how long did you spend on this assignment?
# Set hours equal to a number. You can include decimals if needed.
hours = 8

# Question 1 - Comments (optional)
# You can optionally give me feedback about the number of hours by setting a string below:
q1 = " "

# Question 2
# Which aspects of this assignment did you find most challenging? Were there any significant stumbling blocks?
# Provide your answer in the string below:
q2 = "Implementing the Naive Bayes classification algorithm"

# Question 3
# Which aspects of this assignment did you like? Is there anything you would have changed?
# Provide your answer in the string below:
q3 = "I like the whole assignment, I'm not sure if there is anything I would change."