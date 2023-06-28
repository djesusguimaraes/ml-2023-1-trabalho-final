import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

analyzer = SentimentIntensityAnalyzer()

tweet_label = 'Tweet'
clen_tweet_label = 'Clean Tweet'
sa_label = 'SA'
year_label = 'Year'
date_label = 'Date'


def classify(polarity):
    return "positive" if polarity == 1 else "negative" if polarity == -1 else "neutral"


def extract_tweets():
    df = pd.read_excel('data/2017_01_28 - Trump Tweets.xlsx')
    df = df[[tweet_label, date_label]]
    df[date_label] = pd.to_datetime(df[date_label], format='%m/%d/%Y')
    df[year_label] = df[date_label].dt.year
    return df


def clean_tweet(tweet, stop_words):
    words = word_tokenize(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split()))
    return ' '.join([w for w in words if w.lower() not in stop_words])


def analize_sentiment(tweet):
    polarity = analyzer.polarity_scores(tweet)['compound']

    if polarity >= 0.2:
        return 1

    if polarity <= -0.2:
        return -1

    return 0


def stringfy(tweets, total_len, polarity=0):
    return f"Percentage of {classify(polarity)} tweets: {len(tweets) * 100 / total_len}"


def search_in_tweets(df, search, year=None):
    quantidade = df[clen_tweet_label].str.lower().str.count(search).sum()

    tweets_by_search = df[df[tweet_label].str.lower().str.contains(search)]

    sentiment_by_search = tweets_by_search[sa_label].apply(lambda x: classify(x))

    print(f"Quantidade de vezes que '{search}' apareceu em tweets{f' de {year}' if year else ''}:", quantidade)
    print(f"Sentimentos associados aos tweets mencionando '{search}': {sentiment_by_search.value_counts()}\n")


def search_in_tweets_by_year(df, search, year):
    tweets_from_year = df[df[year_label] == year]

    search_in_tweets(tweets_from_year, search, year)


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')

    stp_words = set(stopwords.words('english'))

    data = extract_tweets()

    # Conta a quantidade de tweets por ano
    tweets_count_by_year = data.groupby(year_label).count()[tweet_label]

    # Cria a coluna de polaridade da análise de sentimentos
    data[clen_tweet_label] = np.array([clean_tweet(tweet, stp_words) for tweet in data[tweet_label]])

    data[sa_label] = np.array([analize_sentiment(tweet) for tweet in data[clen_tweet_label]])

    # Cria uma série temporal para cada ano
    sa_grouped_by_year = data.groupby(year_label)[sa_label]

    def get_mean_by_sentiment(condition):
        return condition.mean() * 100

    positives_by_year_series = sa_grouped_by_year.apply(lambda x: get_mean_by_sentiment(x > 0))
    neutrals_by_year_series = sa_grouped_by_year.apply(lambda x: get_mean_by_sentiment(x == 0))
    negatives_by_year_series = sa_grouped_by_year.apply(lambda x: get_mean_by_sentiment(x < 0))

    positives_by_year_series.plot(color='green', title='Sentiments in tweets by year', label='Positive', legend=True)
    neutrals_by_year_series.plot(color='blue', label='Neutral', legend=True)
    negatives_by_year_series.plot(color='red', label='Negative', legend=True)

    plt.show()

    g_tweets = [tweet for index, tweet in enumerate(data[tweet_label]) if data[sa_label][index] > 0]
    n_tweets = [tweet for index, tweet in enumerate(data[tweet_label]) if data[sa_label][index] == 0]
    b_tweets = [tweet for index, tweet in enumerate(data[tweet_label]) if data[sa_label][index] < 0]

    tweets_len = len(data[tweet_label])

    print(stringfy(b_tweets, tweets_len, -1), stringfy(g_tweets, tweets_len, 1), stringfy(n_tweets, tweets_len))

    search_in_tweets_by_year(data, 'hillary', 2016)
    search_in_tweets_by_year(data, 'mexico', 2017)

    search_in_tweets(data, 'russia')
    search_in_tweets(data, 'putin')
