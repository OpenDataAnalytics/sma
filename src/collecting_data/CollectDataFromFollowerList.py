# -*- coding: utf-8 -*-
"""
@author: Sayali
"""

import tweepy
import json
import time

consumer_key='FJkeW0VV0D6HGPYlF5UfklTK5' 
consumer_secret='IM8zRQFIq4wbKBgikZKNLqiEkHH6ePSg20Ag6bE1QLY6dIQPGM' 
access_token_key='4921031892-twRpm76J6kgd3cWp2d4dIMkp674ocaggbQiUgCX'
access_token_secret='nTPkL7TXTD4winCFu8INTzdE6ALAIYNk9Tb39d4R0DgYS'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
myApi = tweepy.API(auth)

outputFile = open("FollowersTweets_LungAssociation.txt", "a+")

def GetTweets(user_id):
    alltweets = [] 
    try:
        #Collect most recent 200 tweets
        new_tweets = myApi.user_timeline(id=user_id,count=200)
        alltweets.extend(new_tweets)
        #print user_id
         
        #save the id lastest tweet collected
        MAX_ID = alltweets[-1].id - 1
	
        #Collect tweets till last tweets
        while len(new_tweets) > 0:
            new_tweets = myApi.user_timeline(id=user_id, count=200, max_id=MAX_ID)
            alltweets.extend(new_tweets)
            MAX_ID = alltweets[-1].id - 1
   
        for tweet in alltweets:
            outputFile.write(json.dumps(tweet._json) + "\n")
    except:
        pass
        
if __name__ == '__main__':
    inputFile = open("ListOfFollowers_LungAssociation.txt")
    startTime = time.time()
    for line in iter(inputFile):
        GetTweets(line)
    inputFile.close()
    endTime = time.time()
    print endTime - startTime
