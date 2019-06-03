#%% [markdown]
# * Sentiment analysis and topic modeling 
# * Correlation between topic and sentiment
# 
# 1. what are you looking for
# 2. why does it matter
# 3. how do you know you found what you were looking for
# 
# https://developer.twitter.com/en/dashboard

#%% [markdown]
##  Analyzing Tweets on Bryce Harper
### IST736 Text Mining Project
#### Martin Alonso 
#### 2019-03-16

#### Introduction
# The objective of this project is to analyze twitter sentiment towards Bryce Harper. Harper 
# is among the youngest players to reach free agency and one of the best players in Major 
# League Baseball. However, he is also polemical, dividing fan bases opinions on him.  
# During the 2018-2019 offseason, six teams, the Chicago Cubs, Chicago White Sox, 
# Los Angeles Dodgers, Philadelphia Phillies, New York Yankees, San Francisco Giants, and
# Washington Nationals, tried to sign him to a contract, with him finally signing with the 
# Phillies on February 28, 2019.   
# Given how long it took to sign him, and how many teams fought for his services, it would be 
# interesting to discover how the different fan bases reacted to rumors of him signing with
# each team, whether positive or negative. 

#### Methodology
# 327 thousand tweets between November 1, 2018, the day free agency started, and March 3, 2019
# were downloaded using both the Twitter developer API and `twitterscraper` package. Aside 
# from the tweet content, location from where the tweet was sent was acquired and cleaned. 
# These tweets will be grouped by location, and sentiment analysis will be done on these to 
# try and identify how each market feels towards Harper and, finally, try and predict, given
# the sentiment and word usage of the tweet, where the tweet was sent from.  
# The data was first cleaned, as Twitter users are allowed to edit there locations. After locations
# were normalized, the tweets were cleaned using stemming and lemmitization. The data will be
# grouped by city to find any patterns in word usage and word frequency. Then, using Binary
# Vectorizer from the `nltk` package and Support Vector Machines and Naive Bayes from the 
# `sklearn` package, a model will be built to try and predict where the tweets were sent from.

#### Data Munging
# The data has already been acquired prior to starting this model. The code used to obtain
# the tweets will be shown, but the data has already been cleaned and will be loaded using
# `pandas` read_csv function. 

#%%
# Import required packages
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool, cpu_count
from sklearn.feature_selection import chi2
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk import PorterStemmer
from datetime import datetime
from io import StringIO
from PIL import Image
import seaborn as sns 
import json, codecs
import pandas as pd
import numpy as np
import tweepy
import nltk
import time
import re 

sns.set(rc={'figure.figsize':(12,9)})

#%%
# Load files obtained through twitterscraper. This cell does not need to be run again. 
with codecs.open('harper_tweets.json', 'r', 'utf-8') as harper_tweets:
    harper_tweets = json.load(harper_tweets, encoding='utf-8')

#%%
# Extract each element from the tweets and convert them to a pandas DataFrame 
list_tweets = [list(elem.values()) for elem in harper_tweets]
list_columns = list(harper_tweets[0].keys())
tweets = pd.DataFrame(list_tweets, columns=list_columns)

# Export to csv for quick loading later. 
tweets.to_csv('harper_tweets.csv')

#%% [markdown]
# Since these tweets don't have any location information, we'll need to use the `tweepy` package and have access to the Twitter developers API to extract the location from each tweet id. 

#%%
# Consumer and access keys that allow access to the Twitter developer API 
# This cell does not need to be run again and will be commented out. 
#consumer_key = 
#consumer_secret = 

#access_token = 
#access_token_secret = 

# Create the tweepy API 
#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token, access_token_secret)

#api = tweepy.API(auth, wait_on_rate_limit=True)

#%% [markdown]
# The previous cells obtained the Twitter data on Harper. But the data extracted was returned
# as an array. Furthermore, the data does not have the location data that is needed for this
# project.  
# To obtain this data, the `TweetLocation` class was built, in the next cell, and iterated
# over the tweets, returning the location from were each tweet was sent. 

#%%
# %load TweetLocation.py
import tweepy 
import pandas as pd 

class tweet_location:
    """
    Obtains tweet location by passing tweet ids through the parser.
    """

    def __init__(self, id):
        self.id = id

    def get_location(self, id):
        """
        Takes as input a tweet's id. Passes the id through Tweepy's API to obtain the 
        tweet status. Then, the location from were the tweet was sent is extracted, and
        both the tweet id and location are returned. 
        If a tweet does not exist, an error message is returned. 
        """
        try:
            tweet_status = api.get_status(id)
            tweet_location = tweet_status.user.location
            tweet_data = [id, tweet_location]
            return(tweet_data)
        except:
            print("Tweet ID not found")

#%% [markdown]
# Now that the tweet data was downloaded and cleaned, the data will be reloaded and initial 
# EDA will be performed. Namely, we want to know how many tweets were sent per day. We also
# want to know which city sent the most tweets and what the most common words, both positive
# and negative, were sent from the top cities. 

#%%
# Load the tweets that were saved as a csv. Start from this point. 
#tweets = pd.read_csv('harper_tweets.csv', encoding='latin-1')
tweets.head()

#%%
# Import tweet location file. 
tweet_loc = pd.read_csv('harper_tweet_locations.txt', sep=',', encoding='latin-1')
tweet_loc['tweet_id'] = tweet_loc['id'].astype(str)
tweet_loc.head()

#%%
# Merge location file with tweets, obtaining one data set with tweet data and location. 
tweets['id'] = tweets['id'].astype(str)

df_merged = tweets.merge(tweet_loc, left_on='id', right_on='tweet_id', how='left')
df_merged.head()

#%% [markdown]
# Looking at the first five tweets, it is evident that not all tweets have a location. The
# shape of the tweets will be printed, along as the count per city to gauge how many missing
# data there is. 

#%%
# Print number of columns and observations for the dataset. 
print('df_merged shape: {}'.format(df_merged.shape))

# Print tweets by location. 
tweets_by_location = df_merged.groupby('city')['id_x'].count().sort_values(ascending=False)
print(tweets_by_location)

#%% [markdown]
# This present a probelm. There are 327,148 tweets in the data set. However, 207,711 tweets
# (63.5 percent) don't have clean data. 
# However, the cities of interest are well represented, and some additional cleaning will 
# need to be done. For example, there are tweets from New York City, Bronx, Brooklyn, Long
# Island and Staten Island. Though different burroughs of New York City, we can treat these as a single
# location, as they are all homes to Yankee fans. The same can be done with San Francisco, 
# San Jose, and Sacramento, which are considered part of Giants territory.  
#  
# Additionally, Las Vegas is well represented. Though Las Vegas doesn't have a MLB team, it is
# Bryce Harper's hometown, so we'll include it in the dataset. 

#%% 
# Group cities 
new_york = ['new york', 'bronx', 'brooklyn', 'long island', 'staten island', 'queens']
# Though queens is considered Mets territory, we'll add them as they are Philly rivals. 

san_francisco = ['san francisco', 'san jose', 'sacramento', 'oakland']
# Oakland is a different city with its own franchise but it also houses Giant fans

los_angeles = ['los angeles', 'santa monica', 'hollywood', 'long beach', 'huntington beach']
washington_dc = ['washington dc', 'richmond', 'baltimore', 'alexandria']

#%% 
# With the group cities arrays, we'll change the cities in each array to those of the array name
for i in range(len(df_merged)):
    if df_merged.loc[i, 'city'] in new_york:
        df_merged.loc[i, 'city'] = 'new york'
    elif df_merged.loc[i, 'city'] in san_francisco:
        df_merged.loc[i, 'city'] = 'san francisco'
    elif df_merged.loc[i, 'city'] in los_angeles:
        df_merged.loc[i, 'city'] = 'los angeles'
    elif df_merged.loc[i, 'city'] in washington_dc:
        df_merged.loc[i, 'city'] = 'washington dc'

tweets_by_location = df_merged.groupby('city')['id_x'].count().sort_values(ascending=False)
print(tweets_by_location)

#%% [markdown]
# The data has been cleaned a bit more. But given that we want to focus on just a set group
# of cities, we'll restrict the analysis to the first 10 cities listed in the previous list: 
# 1. Philadelphia
# 2. New York
# 3. Chicago
# 4. Washington DC
# 5. Los Angeles
# 6. San Francisco
# 7. San Diego
# 8. St. Louis
# 9. Las Vegas
# 10. Boston

# By keeping this set, the data will drop from 327 thousand to 100,887 observations (31 percent).

#%% 
# Keep observations from previous list 
cities_to_keep = ['philadelphia', 'new york', 'chicago', 'washington dc', 'los angeles', 
                 'san francisco', 'san diego', 'st louis', 'las vegas', 'boston']

# Filter data frame
df_clean = df_merged[df_merged['city'].isin(cities_to_keep)]
df_clean = df_clean.reset_index()
df_clean.head()

#%% [markdown]
# Now that the data has been cleaned, let's start exploring the data set. We want to see how many tweets are sent per day: both total and by city to see if there are any spikes. 

#%%
# Groups tweets by day the tweet was sent 
tweet_count = df_clean[['timestamp', 'id_x', 'city']]
tweet_count['timestamp'] = pd.to_datetime(tweet_count['timestamp']).dt.date
tweet_count = tweet_count.groupby(['timestamp', 'city'])['id_x'].count().reset_index()

#%%
# Plot the data
sns.lineplot(x='timestamp', y='id_x', data=tweet_count)
_ = plt.xticks(rotation=25)
_ = plt.xlabel('Date')
_ = plt.ylabel('Number of tweets')
_ = plt.title('Tweets mentioning Harper per day')
plt.show()

#%% [markdown]
# Not many tweets sent out during the offseason, but there is a massive spike on the day that Harper signed. The same graph will be created, but this time openned by city where the tweet was sent. Hopefully, a pattern will appear.

#%%
# Number of tweets by city 
sns.lineplot(x='timestamp', y='id_x', hue='city', data=tweet_count)
_ = plt.xticks(rotation=25)
_ = plt.xlabel('Date')
_ = plt.ylabel('Number of tweets')
_ = plt.title('Tweets mentioning Harper per day by city')
plt.show()

#%% [markdown]
# Because of that spike, it appears that no city stands out. A third line plot will be 
# created, excluding the last day of the dataset, while a bar graph will be created to gauge 
# the number of tweets sent out by each city on the day Harper signed. 

#%%
# Tweets sent out per day between 2018-11-01 and 2019-02-27
tweet_count2 = tweet_count
tweet_count2.timestamp = pd.to_datetime(tweet_count2['timestamp'])
tweet_count2.set_index(['timestamp'], inplace=True)
tweet_count_02_27 = tweet_count2.loc['2018-11-01':'2019-02-27']

#%%
sns.lineplot(x=tweet_count_02_27.index, y=tweet_count_02_27.iloc[:, 1], hue=tweet_count_02_27.iloc[:, 0], data=tweet_count_02_27)
_ = plt.xticks(rotation=25)
_ = plt.xlabel('Date')
_ = plt.ylabel('Number of tweets')
_ = plt.title('Tweets mentioning Harper per day by city')
plt.show()

#%% [markdown]
# Removing the day that Harper signed, we can see a clearer pattern. There are spikes from Chicago, Los Angeles, and Philadelphia. But the most vocal fans are from Philadelphia, who, by the end of January, had ramped up their tweeting. As the siging date came closer, fans in Los Angeles started tweeting too, while the other teams' fans remained relatively quiet.  
# Now let's see how the tweets were distributed on the actual day. 

#%%
# Needs to be run again 
tweet_count = df_clean[['timestamp', 'id_x', 'city']]
tweet_count['timestamp'] = pd.to_datetime(tweet_count['timestamp']).dt.date
tweet_count = tweet_count.groupby(['timestamp', 'city'])['id_x'].count().reset_index()

#%%
# Obtain tweets after 2019-02-27 
tweet_count3 = tweet_count
tweet_count3.timestamp = pd.to_datetime(tweet_count3['timestamp'])
tweet_count3.set_index(['timestamp'], inplace=True)
tweet_count_02_28 = tweet_count3.loc['2019-02-28':]

#%%
# Create a bar graph with cities on the x-axis and counting the number of tweets
count_by_city = tweet_count_02_28.groupby('city')['id_x'].sum().reset_index()

sns.barplot(x='city', y='id_x', data=count_by_city)
_ = plt.xlabel('City')
_ = plt.ylabel('Number of Tweets')
_ = plt.title('Tweets by City on day Harper signed')
plt.show()

#%% [markdown]
# Over 40,000 tweets sent from Philadelphia on the day that Harper signed! Seems like Philly fans were extactic, while the fans of the other franchises remained relatively quiet. 

#%% [markdown]
#### Data Processing for Sentiment Analysis
# Now that we have seen how the data distributes, we'll start processing the data for text analysis. From the data set that's been worked on, we'll keep the tweet text, location, and tweet date for sentiment analysis. Then, the tweet text will be parsed through, removing special characters and stop words, while also lower-casing the words.  
#
# Once this is done, the text will be passed through stemming and lemming for further cleaning. Then, initial sentiment analysis will be done, highlighting not only positive and negative words but also positive and negative words related to baseball players. This last list has been provided by former Major League Basebal scout Jason Lefkowitz. 
#
# After this, we'll proceed with building some word clouds to highlight both the positive and negative words surrounding Harper. 

#%% 
# Keep timestamp, city, and tweet columns from dataset. 
df_text = df_clean[['timestamp', 'id_x', 'city', 'text']]
df_text.head()
df_text.to_csv('Harper_Tweets_Simp.csv')

#%%
# Convert non-letters to spaces, then remove special characters. 
# Remove special characters and lower case words for easier text analysis
df_text['text'] = df_text['text'].astype(str)
df_text['text'] = df_text['text'].map(lambda x: re.sub(r'\W', ' ', x).lower())
df_text.head()

#%%
# Set stopwords and tokenize text column for stop word removal
stop_words = set(stopwords.words('english'))
df_text['word_tokens'] = df_text.apply(lambda row: word_tokenize(row['text']), axis=1)

# Remove stopwords to filter out noise and reconnect strings into sentences. 
df_text['clean_text'] = df_text['word_tokens'].apply(lambda x: [word for word in x if word not in stop_words])
df_text['clean_text'] = df_text['clean_text'].apply(lambda x: ' '.join(str(word) for word in x))
df_text.head()

#%%
# Initialize stemmer
porter = PorterStemmer()

# Create function that tokenizes words in string and stems words. 
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

#%%
# Stem clean text in df_clean
df_text['stemmed'] = df_text['clean_text'].map(lambda x: stemSentence(x))
df_text.head()

#%% [markdown]
# With the text cleaned and stemmed, it is now time to lemmatize the texts. 

#%%
# Initiate Lemmatizer
lemmatizer = WordNetLemmatizer()

df_text['lemmed'] = df_text['clean_text'].map(lambda x: lemmatizer.lemmatize(x, pos='v'))
df_text.head()

#%% [markdown]
# Now that we have cleaned stemmed and lemmatized texts, we'll analyze text sentiment and print out the most common words featured in positive-minded texts and negative-minded texts. 

#%%
# Initiate sentiment intensity analyzer
sia = SIA()
results = []

for line in df_text['stemmed']:
    pol_score = sia.polarity_scores(line)
    pol_score['stemmed'] = line
    results.append(pol_score)

# Convert results into data frame and print first ten results 
stemmed_sent = pd.DataFrame.from_records(results)

# Add a label to identify whether the text is either positive, negative, or neutral
stemmed_sent['stem_label'] = 0
stemmed_sent.loc[stemmed_sent['compound'] > 0.2, 'stem_label'] = 1
stemmed_sent.loc[stemmed_sent['compound'] < -0.2, 'stem_label'] = -1
stemmed_sent.loc[:10]

#%%
# Now repeat for the lemmed text
results = []

for line in df_text['lemmed']:
    pol_score = sia.polarity_scores(line)
    pol_score['lemmed'] = line
    results.append(pol_score)

# Convert results into data frame and print first ten results 
lemmed_sent = pd.DataFrame.from_records(results)

# Add a label to identify whether the text is either positive, negative, or neutral
lemmed_sent['lem_label'] = 0
lemmed_sent.loc[lemmed_sent['compound'] > 0.2, 'lem_label'] = 1
lemmed_sent.loc[lemmed_sent['compound'] < -0.2, 'lem_label'] = -1
lemmed_sent.loc[:10]

#%% [markdown]
# With the sentiment of the texts now obtained, we'll merge the labels back into the original data set. We'll then print out a bar chart to compare how many tweets we have of each sentiment kind.  
# 
# Once this is done, we'll print out a couple of words clouds to detect which cities loved or hated Harper the most, and what were the alleged reasons behind these thoughts. 

#%%
# Concatenate stemmed and lemmed labels back into the original data frame
df_text2 = pd.concat([df_text, stemmed_sent], axis=1, ignore_index=True)
df_text3 = pd.concat([df_text2, lemmed_sent], axis=1, ignore_index=True)

# Keep columns of interest
df_text3.columns = ['timestamp', 'id', 'city', 'text', 'tokens', 'clean_text', 
                    'stemmed_text', 'lemmed_text', 'compound_stem', 'stem_text', 'neg_stem', 
                    'neu_stem', 'pos_stem',  'stem_label', 'compound_lem', 'lem_text', 
                    'neg_lem', 'neu_lem', 'pos_lem', 'lem_label']

df_text_simp = df_text3[['timestamp', 'id', 'city', 'text', 'clean_text', 'stemmed_text', 
                        'lemmed_text', 'compound_stem', 'stem_label', 'compound_lem', 'lem_label']]

df_text_simp.head()

#%% [markdown]
# Now that we have the sentiment for each phrase, it's time to check how the tweet distribution falls. 

#%%
fig, ax = plt.subplots(figsize=(8, 8))
counts = df_text_simp.stem_label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")
_ = plt.title('Sentiment of Stemmed Words')

plt.show()

#%%
fig, ax = plt.subplots(figsize=(8, 8))
counts = df_text_simp.lem_label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")
_ = plt.title('Sentiment of Lemmatized Words')

plt.show()

#%% [markdown]
# We have two very different stories being told depending on the method we choose. If we use stem words, the majority of tweets are decidedly negative towards Harper, which makes sense if fans of other teams feel spited by him if he decides to sign with another team. However, lemmatized words show that the overall sentiment is neutral, with some slight positive sentiment and almost no negative-sentiment tweets. Given the nature of this project, we'll move on looking only at the stemmed words labels. 
# 
# Before moving on to word clouds, let's see which cities had a more negative sentiment towards Harper and which ones were more welcoming to him. 

#%%
# Group sentiment by city and count number of tweets
city_sent = df_text_simp.groupby(['city', 'stem_label'])['id'].count()
city_sent = city_sent.reset_index()

# Calculate total tweets and percentages for each city 
city_sent_pt = city_sent.pivot(index='city', columns='stem_label', values='id').reset_index()
city_sent_pt.columns = ['city', 'neg', 'neu', 'pos']
city_sent_pt['total'] = city_sent_pt['neg'] + city_sent_pt['neu'] + city_sent_pt['pos']

# Calculate percentage of positive and negative tweets for each city
city_sent_pt['neg_pct'] = city_sent_pt['neg']/city_sent_pt['total']
city_sent_pt['pos_pct'] = city_sent_pt['pos']/city_sent_pt['total']

city_sent_pt

#%%
# Plot negative tweet sentiment by city
fig, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x='city', y='neg_pct', ax=ax, data=city_sent_pt)
_ = plt.xlabel('City')
ax.set_ylabel("Percentage")
_ = plt.title('Percentage of Negative Sentiment Tweets by City')

plt.show()

#%%
# Repeat for positive tweets
fig, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x='city', y='pos_pct', ax=ax, data=city_sent_pt)
_ = plt.xlabel('City')
ax.set_ylabel("Percentage")
_ = plt.title('Percentage of Positive Sentiment Tweets by City')

plt.show()

#%% [markdown]
# These results are very interesting. One would have expected the Philadelphia fan base to be pretty excited about acquiring a player like Harper. However, we can see that they are very negative towards him, something that the baseball media, and former Philadelphia Phillies players, have warned him might happen. We'll have to explore what is it that drives Philadelphia fans to be so negative about Harper while other fan bases are rather positive about him. 

#%%
def process_text(headlines):
    tokens = []
    for line in headlines:
        toks = word_tokenize(line)
        toks = [t for t in toks if t not in stop_words]
        tokens.extend(toks)
    
    return tokens

#%%
# Print out the top 20 negative words from Philadelphia 
neg_phillies = list(df_text_simp[(df_text_simp.stem_label==-1) & (df_text_simp.city=='philadelphia')].stemmed_text)

neg_tokens = process_text(neg_phillies)
neg_freq = nltk.FreqDist(neg_tokens)

neg_freq.most_common(25)

#%%
pos_rest = list(df_text_simp[(df_text_simp.stem_label==1) & (df_text_simp.city!='philadelphia')].stemmed_text)

pos_tokens = process_text(pos_rest)
pos_freq = nltk.FreqDist(pos_tokens)

pos_freq.most_common(25)

#%% [markdown]
# Now that we have the sentiment by city for Harper, let's build word clouds to compare which words drive Philly fans from other fans.

#%%
# First, let's build a word cloud for negative sentiment tweets from Philadelphia
neg_phi = df_text_simp[(df_text_simp.stem_label==-1) & (df_text_simp.city=='philadelphia')].stemmed_text
text = " ".join(tweet for tweet in neg_phi)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#%%
# Let's build a second word cloud to compare the rest of the fan bases 
neg_rest = df_text_simp[(df_text_simp.stem_label==-1) & (df_text_simp.city!='philadelphia')].stemmed_text
text = " ".join(tweet for tweet in neg_rest)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% [markdown]
# What about positive sentiment? 

#%%
# Positive sentiment from Philadelphia 
pos_phi = df_text_simp[(df_text_simp.stem_label==1) & (df_text_simp.city=='philadelphia')].stemmed_text
text = " ".join(tweet for tweet in pos_phi)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#%%
# And positive sentiment from the rest 
pos_rest = df_text_simp[(df_text_simp.stem_label==1) & (df_text_simp.city!='philadelphia')].stemmed_text
text = " ".join(tweet for tweet in pos_rest)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#%% [markdown]
# Finally, we want to see how positive and negative sentiment has trended over time for these fan bases. 

#%%
# First, convert timestamp to datetime
df_text_simp['timestamp'] = pd.to_datetime(df_text_simp.timestamp).dt.date

# Group observations by timestamp, city, and sentiment
time_sent = df_text_simp.groupby(['timestamp', 'city', 'stem_label'])['id'].count().reset_index()

# Pivot table to calculate percentages 
time_sent_pt = time_sent.set_index(['timestamp', 'city', 'stem_label']).unstack()
time_sent_pt = time_sent_pt.reset_index()
time_sent_pt.columns = ['timestamp', 'city', 'neg', 'neu', 'pos']
time_sent_pt = time_sent_pt.fillna(0)
time_sent_pt.head()

#%%
time_sent_pt['cum_neg'] = time_sent_pt.groupby('city')['neg'].cumsum()
time_sent_pt['cum_neu'] = time_sent_pt.groupby('city')['neu'].cumsum()
time_sent_pt['cum_pos'] = time_sent_pt.groupby('city')['pos'].cumsum()
time_sent_pt.head()

#%%
# Create total, pos_pct, and neg_pct columns
time_sent_pt['total'] = time_sent_pt['cum_neg']+time_sent_pt['cum_neu']+time_sent_pt['cum_pos']
time_sent_pt['neg_pct'] = time_sent_pt['cum_neg']/time_sent_pt['total']
time_sent_pt['pos_pct'] = time_sent_pt['cum_pos']/time_sent_pt['total']

time_sent_pt.head()

#%%
# Plot positive sentiment percentage by city
sns.lineplot(x='timestamp', y='pos_pct', hue='city', data=time_sent_pt)
plt.show()

#%%
sns.lineplot(x='timestamp', y='neg_pct', hue='city', data=time_sent_pt)
_ = plt.xlabel('Date')
_ = plt.xticks(rotation=75)
_ = plt.ylabel('Negative Sentiment')
_ = plt.title('Negative Sentiment by Date by City')
_ = plt.legend(loc = 'upper left')
plt.show()

#%% [markdown]
# This is really interesting. Contrary to what one may think, the Philadelphia Phillies fan base was not as excited obout getting Bryce Harper as one would think. Perhaps they are worried about his contract and him not living up to their expectiations but we can see that, until the moment he signed, Phillies fans where basically indifferent towards Harper, same as the other teams' fanbases.
# 
#
#### Data Modeling
# With all this information, we'll now turn to building a model capable of predicting whether we can determine the city and based on the wording and sentiment of the tweets sent out. For this, we'll use both unigrams and bigrams within a count vectorizer. We'll pass the vectorizer through a Support Vector Machine (SVM); training the model with 80 percent of the data, chosen at random, while testing on the remaining 20 percent. 
#
# Before we start predicting which tweet belongs to which city, we must first convert cities to numeric categories. The cities will be label-encoded. 

#%%
# Select columns that will be used in the model 
cols = ['city', 'stemmed_text', 'stem_label']
df_model = df_text_simp[cols]
df_model.head()

#%%
# Label encode the city column
df_model['city_id'] = df_model['city'].factorize()[0]
city_id_df = df_model[['city', 'city_id']].drop_duplicates().sort_values('city_id')
city_to_id = dict(city_id_df.values)
id_to_city = dict(city_id_df[['city_id', 'city']].values)
df_model.head()

#%%
# Build TF-IDF and Count Vectorizers for the models. Keep settings as close as possible to make comparable.
#tfidf = TfidfVectorizer(min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words='english')

# Fit the vectorizers to the data
#tf_fit = tfidf.fit_transform(df_model.stemmed_text)

# Select the labels 
#labels = df_model.city_id
#print(tf_fit.shape)

#%%
# Build the CountVectorizer
#cv = CountVectorizer(min_df=5, binary=True, encoding='utf-8', ngram_range=(1,2), stop_words='english')
#cv_fit = cv.fit_transform(df_model.stemmed_text)
#print(cv_fit.shape)

#%%
# Find terms that are most correlated with each city. 
#N = 2
#for city, city_id in sorted(city_to_id.items()):
#  tfidf_chi2 = chi2(tf_fit, labels == city_id)
#  tfidf_indices = np.argsort(tfidf_chi2[0])
#  tfidf_names = np.array(tfidf.get_feature_names())[tfidf_indices]
#  tf_unigrams = [v for v in tfidf_names if len(v.split(' '))==1]
#  tf_bigrams = [v for v in tfidf_names if len(v.split(' '))==2]
#  print("# '{}':".format(city))
#  print("  . Most correlated tfidf unigrams:\n. {}".format('\n. '.join(tf_unigrams[-N:])))
#  print("  . Most correlated tfidf bigrams:\n. {}".format('\n. '.join(tf_bigrams[-N:])))
  

#%%
# Build the first model using stemmed text
#X_train, X_test, y_train, y_test = train_test_split(df_model['stemmed_text'], df_model['city'], random_state = 3, test_size=.2)

# Transforms the text using the previously created vectorizers
#X_train_cv = cv.fit_transform(X_train)
#X_train_tfidf = tfidf.fit_transform(X_train)

#%% [markdown]
# We'll first build a multinomial Naive Bayes model and test it with some newly created tweets. After this, we'll build a SVM model using five-fold cross-validation. 

# Creote the Multinomial Naive Bayes model for the Count and TF-IDF Vectorizers
#cv_mnb = MultinomialNB().fit(X_train_cv, y_train)
#tfidf_mnb = MultinomialNB().fit(X_train_tfidf, y_train)

#%%
print(cv_mnb.predict(cv.transform(["Harper shouldn't sign with the Phillies! He's terrible."])))
print(cv_mnb.predict(cv.transform(["13 years $330 million?!?!?!. Steinbrenner's could've easily topped that."])))
print(cv_mnb.predict(cv.transform(["Bring Bryce to Hollywood!"])))
print(cv_mnb.predict(cv.transform(["F@ck him! He doesn't deserve that money! Very inconsistent player."])))

#%%
print(tfidf_mnb.predict(tfidf.transform(["Harper shouldn't sign with the Phillies! He's terrible."])))
print(tfidf_mnb.predict(tfidf.transform(["13 years $330 million?!?!?!. Steinbrenner's could've easily topped that."])))
print(tfidf_mnb.predict(tfidf.transform(["Bring Bryce to Hollywood!"])))
print(tfidf_mnb.predict(tfidf.transform(["F@ck him! He doesn't deserve that money! Very inconsistent player."])))

#%% [markdown]
# Not bad! Seems like the Count Vectorizer does a better job at identifying likeliest place the tweet has come from. 
#
# Let's build an SVM now to compare how the model runs 

#%%
svm = LinearSVC(dual=True, C=1.0, class_weight=None, penalty='l2', random_state=3, max_iter=5000)

# Build Count Vectorizer pipeline
cv_clf = svm.fit(X_train_cv, y_train)
cv_svm = cross_val_score(cv_clf, X_train_cv, y_train, cv=5)

#%%
# Build TF-IDF Vectorizer pipeline
tfidf_clf = svm.fit(X_train_tfidf, y_train)
tfidf_svm = cross_val_score(tfidf_clf, X_train_tfidf, y_train, cv=5)

#%%
print(cv_clf.predict(cv.transform(["Harper shouldn't sign with the Phillies! He's terrible."])))
print(cv_clf.predict(cv.transform(["13 years $330 million?!?!?!. Steinbrenner's could've easily topped that."])))
print(cv_clf.predict(cv.transform(["Bring Bryce to Hollywood!"])))
print(cv_clf.predict(cv.transform(["F@ck him! He doesn't deserve that money! Very inconsistent player."])))

#%%
print(tfidf_clf.predict(tfidf.transform(["Harper shouldn't sign with the Phillies! He's terrible."])))
print(tfidf_clf.predict(tfidf.transform(["13 years $330 million?!?!?!. Steinbrenner's could've easily topped that."])))
print(tfidf_clf.predict(tfidf.transform(["Bring Bryce to Hollywood!"])))
print(tfidf_clf.predict(tfidf.transform(["F@ck him! He doesn't deserve that money! Very inconsistent player."])))


#%% [markdown]
# 1. predict
# 2. cross matrix
# 3. model comparison
# 4. conclusions
# 5. change percentage senitment graph to cumulative 