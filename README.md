### Natural-Language-Processing

In this assignment, I applied natural language processing to understand the sentiment in the latest news articles featuring Bitcoin and Ethereum. Also I applied fundamental NLP techniques to better understand the other factors involved with the coin prices such as common words and phrases and organizations and entities mentioned in the articles.

Completed the following tasks:

1. [Sentiment Analysis](#Sentiment-Analysis)
2. [Natural Language Processing](#Natural-Language-Processing)
3. [Named Entity Recognition](#Named-Entity-Recognition)

---

### Files

Starter_Code/crypto_sentiment.ipynb

---

#### Sentiment Analysis

Use the [newsapi](https://newsapi.org/)  pulled the latest news articles for Bitcoin and Ethereum and created a DataFrame of sentiment scores for each coin.

Used descriptive statistics to answer the following questions:

> Which coin had the highest mean positive score? - Bitcoin

> Which coin had the highest negative score? -  Ethereum

> Which coin had the highest positive score? - Ethereum

#### Natural Language Processing

In this section, I used NLTK and Python to tokenize the text for each coin and looked at the ngrams and word frequency for each coin.

1. Used NLTK to produce the ngrams for N = 2.
2. Listed the top 10 words for each coin.

Finally, generated word clouds for each coin to summarize the news for each coin.

![btc-word-cloud.png](Images/btc-word-cloud.png)

![eth-word-cloud.png](Images/eth-word-cloud.png)

#### Named Entity Recognition

In this section, I will built a named entity recognition model for both coins and visualize the tags using SpaCy.
