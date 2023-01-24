import tweepy as tw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

consumer_key = "gf8UgoVobK0GrAULQ0FIYTGia" #API key
consumer_secret = 'EwQsrWxNuttEXMCZQnt44Ii4rWyJFHW6RlCKneQm' #API Key Secret
access_token = '1617063469693472768-AHylq6tzcweWtZvtro83tFuLXaI9NV'
access_token_secret = 'jFKQpW73cEys2SYVPDLEIPgr8u7tfm54o4HO0NueujWDO'
baerer_token = 'AAAAAAAAAAAAAAAAAAAAALh4lQEAAAAARqsvawn%2B3ked68zFk3G3D1kjLmA%3DdiHxtnnj3GwnnPLxwqBk4bwOzfHgFsZBU6t2H2ivPDZT8laJ6i'
#Authenticate
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

hashtag = '#avatar'
query = tw.Cursor(api.search_tweets, q=hashtag).items(1000)
tweets = [{'Tweets': tweet.text, 'Timestamp': tweet.created_at} for tweet in query]
print(tweets)

