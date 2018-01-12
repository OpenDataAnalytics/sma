# -*- coding: utf-8 -*-
# created by Jie Liu
import io
import re

from sklearn import svm, model_selection, tree
import json
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time


stopwords = ['or', 'could', 'should', 'along', 'nor', 'nothing', 'may', 'etc', 'show', 'them', 'whereby', 'mostly', 'nine', 'system', 'found', 'nobody', 'only', 'hundred', 'it', 'beyond', 'over', 're', 'sometime', 'thence', 'up', 'else', 'both', 'take', 'must', 'full', 'find', 'since', 'how', 'third', 'those', 'upon', 'thereafter', 'empty', 'herein', 'sixty', 'than', 'other', 'interest', 'whose', 'below', 'four', 'is', 'these', 'cant', 'all', 'alone', 'always', 'at', 'someone', 'first', 'done', 'less', 'next', 'off', 'nevertheless', 'part', 'front', 'whereafter', 'eight', 'six', 'something', 'to', 'are', 'amoungst', 'fifty', 'further', 'would', 'onto', 'amount', 'detail', 'well', 'even', 'twelve', 'meanwhile', 'whenever', 'until', 'thru','seemed', 'whatever', 'although', 'otherwise', 'anything', 'that', 'hence', 'towards', 'between', 'here', 'co', 'any', 'back', 'every', 'might', 'besides', 'ten', 'were', 'because', 'some', 'ever', 'and', 'for', 'through', 'move', 'formerly', 'keep', 'none', 'inc', 'once', 'hereby', 'never', 'then', 'but', 'cannot', 'elsewhere', 'everywhere', 'however', 'forty', 'five', 'namely', 'enough', 'get', 'whither', 'an', 'with', 'mill', 'another', 'they', 'above', 'whole', 'please', 'latterly', 'sometimes', 'under', 'whence', 'describe', 'bottom', 'made', 'do', 'seem', 'latter', 'via', 'what', 'toward', 'anyhow', 'name', 'though', 'several', 'also', 'least', 'put', 'noone', 'therefore', 'in', 'almost', 'again', 'fifteen', 'so', 'see', 'anywhere', 'there', 'throughout', 'top', 'without', 'becomes', 'often', 'seeming', 'why', 'thus', 'indeed', 'thereby', 'of', 'bill', 'into', 'have', 'amongst', 'sincere', 'after', 'already', 'such', 'either', 'wherein', 'de', 'nowhere', 'becoming', 'from', 'everything', 'moreover', 'among', 'about', 'yet', 'per', 'as', 'no', 'afterwards', 'others', 'during', 'being', 'if', 'when', 'beforehand', 'across', 'by', 'seems', 'beside', 'call', 'con', 'couldnt', 'ie', 'before', 'un', 'behind', 'eleven', 'still', 'therein', 'hereafter', 'whether', 'not', 'perhaps', 'can', 'anyway', 'thereupon', 'many', 'own', 'where', 'whereupon', 'within', 'wherever', 'down', 'few', 'most', 'around', 'much', 'somewhere', 'which', 'each', 'while', 'yourselves', 'out', 'her', 'hers', 'too', 'together', 'hereupon', 'yours', 'rather', 'very', 'side', 'somehow', 'neither', 'on', 'eg', 'whereas', 'ltd', 'more', 'except', 'twenty', 'go', 'fill', 'former', 'last',
             'going', 'too.', '...', '-', '&', '+']



# true asthma tweets must related to the author
personal_pronouns = ["i", "i'm", "i've", "me", "we", "my"]

# true asthma tweets must mentioned asthma
key_word = "asthma"

# true asthma tweets must have supporting symptoms
key_symptoms = ["inhaler", "wheez", "sneez", "breath", "surviv", "fever", "cough", "lung", "throat", "chest", "stress",
             "headache", "respirat", "weak", "tired", "weak", "anxiety", "panic", "pale", "moody", "cold", "exercise",
             "sweaty", "blue", "lips", "fingernail", "stress", "irritant", "parent", "famil", "pollen", "air", "albuterol",
             "xolair", "montelukast", "nebulizer", "flovent", "singulair", "advair", "bronchodilator"]

# tweet phrase are very useful features
# task 1
# \W represent any Any non-alphanumeric character (equivalent to [^a-zA-Z0-9_].
# * represent Zero or more of previous item
# (A|B) represent A or B
key_phrase = ["my\W*asthma", "asthma\W*attack", "i\W*have\W*asthma", "i’ve\W*had\W*asthma", "have\W*asthma", "have\W*a\W*asthma",
              "have\W*#asthma", "have\W*severe\W*asthma", "have\W*bad\W*asthma", "have\W*slight\W*asthma", "i\W*had\W*asthma", "had\W*asthma",
              "had\W*a\W*asthma", "had\W*severe\W*asthma", "having\W*asthma", "i\W*got\W*asthma", "got\W*asthma", "get\W*asthma",
              "asthma\W*got\W*me", "getting\W*asthma", "have\W*chronic\W*asthma", "have\W*skin\W*asthma", "caught\W*asthma",
              "an\W*asthmatic", "fighting\W*with\W*asthma", "surfer\W*from\W*asthma", "with\W*asthma", "of\W*asthma", "like\W*asthma",
              "this\W*asthma", "the\W*asthma", "got\W*bronchospasm", "had\W*a\W*bronchospasm", "had\W*bronchospasm", "an\W*asthma", "fuck\W*asthma",
              "(trigger|triggered)\W*asthma", "induced\W*asthma", "asthma\W*trigger", "cough\W*(and|&)\W*asthma", "allergies\W*(and|&)\W*asthma",
              "stress\W*(and|&)\W*asthma", "cough\W*asthma", "asthma\W*(medication|meds)", "asthma\W*treatment", "status\W*asthmaticus",
              "asthma\W*control", "(causes|cause)\W*asthma", "my\W*inhaler", "i\W*(can not|can’t)\W*breathe"]

# number of hashtags may be useful features
key_hashtags = ["#asthmasurvivor", "#asthma", "#BeatingAsthma"]


# air condition words may be useful features
air_condition = ["poll", "air", "weather", "co", "no2", "pm2.5", "cold", "dry", "humid", "moist", "hot"]

# number of features we added: 103

def prepare_data(file1, file2):
    # load positive and negative tweets
    positive_tweets = []
    negative_tweets = []

    for line in open(file1).readlines():
        text = line.lower().strip()
        positive_tweets.append([int(1), text])

    for line in open(file2).readlines():
        text = line.lower().strip()
        negative_tweets.append([int(0), text])

    tweets = positive_tweets + negative_tweets


    # Extract the vocabulary of keywords
    vocab = dict()
    for class_label, text in tweets:
        for term in text.split():
            term = term.lower()
            if len(term) >= 1 and term not in stopwords:
                if term in vocab:
                    vocab[term] = vocab[term] + 1
                else:
                    vocab[term] = 1

    # Remove terms whose frequencies are less than a threshold (e.g., 15)
    vocab = {term: freq for term, freq in vocab.items() if freq > 30}

    # Generate an id (starting from 0) for each term in vocab
    vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
    #print(vocab)  # number of features =


    # Generate X and y
    X = []
    y = []

    for class_label, text in tweets:
        x = [0] * (len(vocab) + len(key_symptoms) + len(key_phrase) + len(air_condition) + 3)

        # compute the vocabulary features
        terms = [term for term in text.split() if len(term) >= 1]
        for term in terms:
            if term in vocab:
                x[vocab[term]] += 1

            # compute the personal_pronoun feature
            if term in personal_pronouns:
                x[len(vocab) + 0] += 1

        # compute key word asthma feature
        x[len(vocab) + 1] += text.count(key_word)

        # compute the frequency of key hashtag feature
        for item in key_hashtags:
            x[len(vocab) + 2] += text.count(item)

        symptom_count = 1
        for item in key_symptoms:
            x[len(vocab) + 2 + symptom_count] += text.count(item)
            symptom_count += 1

        phrase_count = 1
        for item in key_phrase:
            x[len(vocab) + 2+ len(key_symptoms) + phrase_count] += text.count(item)
            phrase_count += 1

        air_count = 1
        for item in air_condition:
            x[len(vocab) + 2 + len(key_symptoms) +len(key_phrase) + air_count] += text.count(item)
            air_count += 1

        y.append(class_label)
        X.append(x)

    return X, y



def svm_classification():

    X, y = prepare_data("./data/positive_tweets_new.txt", "./data/negative_tweets_new.txt")

    # 10 folder cross validation to estimate the best w and b, using SVM
    clf = svm.SVC(kernel='linear', C= 1, class_weight={1: 2})
    scores = cross_val_score(clf, X, y, cv = 10, scoring = make_scorer(f1_score))
    print('f1_macro: ', "mean: ", np.mean(scores), "std: ", np.std(scores))

    scores = cross_val_score(clf, X, y, cv = 10, scoring='precision')
    print('precision: ', "mean: ", np.mean(scores), "std: ", np.std(scores))

    scores = cross_val_score(clf, X, y, cv = 10, scoring='recall')
    print('recall: ', "mean: ", np.mean(scores), "std: ", np.std(scores))

svm_classification()












