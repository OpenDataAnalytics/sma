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

class StreamListener(tweepy.StreamListener):
    def on_data(self, raw_data):
        try:
            jdata = json.loads(str(raw_data))
            outputFile = open("Asthma_Stream_Data.txt", "a+")
            outputFile.write(json.dumps(jdata) + "\n")
            outputFile.close()
        except:
            print 'Data writting exception.'

def CollectStreamData():
    while(True): 
        sl = StreamListener()
        stream = tweepy.Stream(auth, sl)
        try: 
            stream.filter(track = ["asthma","asthma attack","inhaler","wheezing","sneezing","runny nose","albuterol","xolair","montelukast","nebulizer","flovent","singulair","advair","bronchodilator","bronchodilators","short of breath","chest tight","difficulty breathing","trouble breathing"])
        except:
            print 'Exception occur!'
            
def CollectRestData():
    query = "asthma" or "asthma attack" or "inhaler" or "wheezing" or "sneezing" or "runny nose" or "albuterol" or "xolair" or "montelukast" or "nebulizer" or "flovent" or "singulair" or "advair" or "bronchodilator" or "bronchodilators" or "short of breath" or "chest tight" or "difficulty breathing" or "trouble breathing" 
    GEO = "40.7127750,-74.0059730,30mi" #NYC 
    outputFile = open("Asthma_Rest_Data.txt", "a+")
    
    #Collect most recent 100 tweets
    tweets = myApi.search(q=query, geocode=GEO, count=100)
    for tweet in tweets:
        outputFile.write(json.dumps(tweet._json) + "\n")
    
    MAX_ID = tweets[-1].id
    
    #Continue collecting tweets till last tweet    
    while len(tweets) > 0:
        try:
            tweets = myApi.search(q=query, geocode=GEO, count=100, max_id = MAX_ID)
            if tweets:
                MAX_ID = tweets[-1].id
                print MAX_ID, len(tweets)
                for tweet in tweets:
                    outputFile.write(json.dumps(tweet._json) + "\n")
    
        except tweepy.TweepError:
            print('exception raised, waiting for 15 minutes')
            time.sleep(10*60)
            break
              
if __name__ == '__main__':
    #Collect tweets using Stream API
    CollectStreamData()
    
    #Collect tweets using REST API
    #CollectRestData()



