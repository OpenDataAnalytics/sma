# -*- coding: utf-8 -*-
"""
@author: Sayali
"""

import tweepy
import time

consumer_key='FJkeW0VV0D6HGPYlF5UfklTK5' 
consumer_secret='IM8zRQFIq4wbKBgikZKNLqiEkHH6ePSg20Ag6bE1QLY6dIQPGM' 
access_token_key='4921031892-twRpm76J6kgd3cWp2d4dIMkp674ocaggbQiUgCX'
access_token_secret='nTPkL7TXTD4winCFu8INTzdE6ALAIYNk9Tb39d4R0DgYS'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
myApi = tweepy.API(auth)

#Added 60 sec delay when cursor raise RateLimitError
def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(1 * 60)

if __name__ == '__main__': 
	#List of twitter accounts whose follower id's needs to be collected
    accounts = ['NYCAllergyDr', 'LungAssociation', 'ACAAI', 'AAFANational'] 
  
    for account in accounts:
		startTime = time.time()
		fileName = "ListOfFollowers_" + account + ".txt"
		print fileName
		outputFile = open(fileName, "a+")
		print account
		
		#Collect follower ID's
		for follower in limit_handled(tweepy.Cursor(myApi.followers, id=account).items()):
			outputFile.write(follower.id_str + '\n')
		endTime = time.time()
		print endTime - startTime