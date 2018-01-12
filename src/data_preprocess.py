# -*- coding: utf-8 -*-
# created by Jie Liu
import re
from datetime import datetime

from sklearn import svm, model_selection, tree
import json
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time

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


# task 2: find tweets have one of the phrase

def select_phrase_tweet_1(file):
    count = 0
    for line in open(file).readlines():
        text = line.lower().strip()
        for item in key_phrase:
            if re.search(item, text):
                with open("./data/tweets_with_phrase.txt", 'a') as f:
                    f.write(json.dumps(text))
                    f.write('\n')
                    count += 1
                break
    print("number of tweets with phrase in this file"+file+": ", count)

select_phrase_tweet_1("./data/positive_tweets.txt")

def select_phrase_tweet_2(file):
    count = 0
    for line in open(file).readlines():
        tweet = json.loads(line)
        text = tweet['text'].lower().strip()
        for item in key_phrase:
            if re.search(item, text):
                with open("./data/tweets_with_phrase.txt", 'a') as f:
                    f.write(json.dumps(text))
                    f.write('\n')
                    count += 1
                break
    print("number of tweets with phrase in this file" + file + ": ", count)


select_phrase_tweet_2('../Asthma/Data/FollowersTweets_CDCasthma.txt')
select_phrase_tweet_2('../Asthma/Data/FollowersTweets_ACAAI.txt')
select_phrase_tweet_2('../Asthma/Data/FollowersTweets_AAFANational.txt')












