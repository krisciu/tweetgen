
import tweepy as tweepy
import os


# Authenticate to Twitter
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')
bearer_token = os.getenv('BEARER_TOKEN')


client = tweepy.Client(bearer_token,consumer_key,consumer_secret,access_token,access_token_secret)


f = open("testtext.txt", "r")

response = client.create_tweet(text=f.read())
print(response)