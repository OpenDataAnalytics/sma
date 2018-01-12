# -*- coding: utf-8 -*-

import tweepy
import time

consumer_key='FJkeW0VV0D6HGPYlF5UfklTK5' 
consumer_secret='IM8zRQFIq4wbKBgikZKNLqiEkHH6ePSg20Ag6bE1QLY6dIQPGM' 
access_token_key='4921031892-twRpm76J6kgd3cWp2d4dIMkp674ocaggbQiUgCX'
access_token_secret='nTPkL7TXTD4winCFu8INTzdE6ALAIYNk9Tb39d4R0DgYS'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
myApi = tweepy.API(auth)

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(1 * 60)

if __name__ == '__main__':    
    outputFile = open("ListOfFollowers_Asthma3Ways.txt", "a+")
    accounts = ['Asthma3Ways'] 
#['CDCasthma', 'AAFANational', 'ACAAI', 'LungAssociation']
  
    startTime = time.time()
    for account in accounts:
        print account
        for follower in limit_handled(tweepy.Cursor(myApi.followers, id=account).items()):
            outputFile.write(follower.id_str + '\n')
        endTime = time.time()
        print endTime - startTime
