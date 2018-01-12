# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:12:46 2017

@author: Sayali
"""

import requests
from bs4 import BeautifulSoup

outputFile = open("FollowersCount_LungAssociation1.txt", "a+")
inputFile = open('FCount_LungAssociation.txt')
 
line = inputFile.readline()

while line:
    try:
        val = line.split(",")
        screenName = val[0]
        id = val[1].replace("\n","")
    
        url = "https://twitter.com/{}".format(screenName)
        soup =BeautifulSoup(requests.get(url).text, 'html.parser')
        profile_views=  soup.find('li', {'class':'ProfileNav-item--followers'}).find('span', {'class':'ProfileNav-value'}).text
        
        t = "%s,%s,%s" % (screenName, id, str(profile_views).replace(",", ""))
        outputFile.write(t + '\n')
        
        line = inputFile.readline()
    
    except:
        pass 
            