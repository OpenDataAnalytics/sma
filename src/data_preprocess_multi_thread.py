# -*- coding: utf-8 -*-
# created by Adil Alim
import re
from datetime import datetime

from sklearn import svm, model_selection, tree
import json
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import os
import multiprocessing
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
              "asthma\W*control", "(causes|cause)\W*asthma", "my\W*inhaler", "i\W*(can not|can’t)\W*breathe",
              "asthma","asthma attack","inhaler","wheezing","sneezing","runny nose","albuterol","xolair","montelukast","nebulizer",
              "flovent","singulair","advair","bronchodilator","bronchodilators","short of breath","chest tight","difficulty breathing",
              "trouble breathing","has asthma","having asthma attack","had an asthma",
              "I develop asthma","cough and asthma","getting out of breath",
              "couldn\'t breath","had a bronchospasm" ,"having an asthma attack",
              "so sick","suffered chronic asthma","coughing",
              "i got bronchospasm","tightness in neck and chest",
              "I got asthma","my asthma","heavy breathing",
              "terrible anxiety","was asthmatic","respiratory illness due to pollution",
              "my respiratory illness","sort of respiratory illness"]


personal_pronouns = ["i", "i'm", "i've", "me", "my"]
# task 2: find tweets have one of the phrase

def check_appear_personal_pron(text):
    flag = False
    terms = text.split()
    for str in personal_pronouns:
        if str in terms:
            flag = True
            break
    return flag

def relevance_check(line):
    tweet = json.loads(line)
    text = tweet['text'].lower().strip()
    for item in key_phrase:
        if re.search(item, text) and check_appear_personal_pron(text):
            return True
    return False


class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means we should exit
                # print '%s: Exiting' % proc_name
                break
            # print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.result_queue.put(answer)
        return


class Task(object):
    def __init__(self, item):
        self.item = item
    def __call__(self):
        # this is the place to do your work
        if relevance_check(self.item):
            return self.item
        else:
            return None
    def __str__(self):
        return '%s' % (self.item)

if __name__ == '__main__':
    
    #FollowersTweets_CDCasthma.txt, FollowersTweets_ACAAI.txt and FollowersTweets_AAFANational.txt
    # CDCasthma  ACAAI  AAFANational
    data_name="AAFANational"
    in_file = '/home/apdm02/data/Data/FollowersTweets_'+data_name+'.txt'
    out_file = '/home/apdm02/workspace/NIH/data/tweets_with_phrase_'+data_name+'.txt'

    # Establish communication queues
    tasks = multiprocessing.Queue()
    
    results = multiprocessing.Queue()
    # Start consumers
    num_consumers = 8 # We only use 5 cores.
    print 'Creating %d consumers' % num_consumers
    consumers = [ Consumer(tasks, results)
                  for i in xrange(num_consumers) ]
    for w in consumers:
        w.start()
    
    lines = open(in_file).readlines()
    num_jobs = len(lines)
    total_tweet=num_jobs
    # Enqueue jobs
    for line in lines:
        tasks.put(Task(line))
            
    # Add a poison pill for each consumer
    for i in xrange(num_consumers):
        tasks.put(None)

    relevance_lines = []
    fin = 0
    # Start printing results
    while num_jobs:
        result = results.get()
        if result is not None:
            relevance_lines.append(result)
        num_jobs -= 1

    if os.path.exists(out_file):
        os.remove(out_file)

    count = 0
    for line in relevance_lines:
        with open(out_file, 'a') as f:
            f.write(line)
            #f.write('\n')
            count += 1
    print("number of tweets with phrase in this file" + in_file + ": ", count,"/",total_tweet)
