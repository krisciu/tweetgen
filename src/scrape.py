import snscrape.modules.twitter as sntwitter
import pandas as pandas


tweets_list = []

for tweet in sntwitter.TwitterSearchScraper('from:SCHIZO_FREQ').get_items():
	print(tweet.content)
	tweets_list.append([tweet.content])
# Creating a dataframe from the tweets list above 
tweets_df1 = pd.DataFrame(tweets_list, columns=['Text'])

tweets_df1.to_csv(index=False)