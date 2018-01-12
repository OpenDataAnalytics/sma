# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:01:10 2017

@author: sayal
"""

import json

if __name__ == '__main__':
    hashTagVocab = dict()  
    files = ['AAFANational'] #, 'AAFANational', 'ACAAI', 'AllergyReliefNY'
    
    line_old = ''
    screenName_old = ''
    count_old = ''
    for file in files:
        fileName = "FollowersTweets_" +  file + ".txt"
        print fileName
        outputFileName = "FollowersCount_" +  file + ".txt"
        print outputFileName
        
        outputFile = open(outputFileName, "a+")
        with open(fileName, 'r') as f:
            for line in f:
                tweet = json.loads(line) # load it as Python dict
                if line_old == '':
                    line_old = line
                    id_old = tweet['user']['id_str']
                    screenName_old = tweet['user']['screen_name']
                    count_old = tweet['user']['followers_count']
                if id_old == tweet['user']['id_str']:
                    continue
                else:              
                    t = "%s,%s,%s" % (screenName_old, id_old, count_old)
                    outputFile.write(t + '\n')
                    line_old = line
                    id_old = tweet['user']['id_str']
                    screenName_old = tweet['user']['screen_name']
                    count_old = tweet['user']['followers_count']
        t = "%s,%s,%s" % (screenName_old, id_old, count_old)
        outputFile.write(t + '\n')
                
        

            
                

    