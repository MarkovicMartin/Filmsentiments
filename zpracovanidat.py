import pandas as pd
import numpy as np
import syllables
from nltk.stem import WordNetLemmatizer
from nltk import tokenize, ngrams
from nltk.corpus import stopwords
import re
import textstat
import itertools
import collections
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
https://www.analyticsvidhya.com/blog/2022/09/complete-guide-to-analyzing-movie-reviews-using-nlp/
df = pd.read_csv('out.csv')


def clean_text(a):
    a_replaced = re.sub('[^A-Za-z0-9]+', ' ', a)
    a_replaced = re.sub(r'w+:/{2}[dw-]+(.[dw-]+)*(?:(?:/[^s/]*))*', '', a_replaced)
    a_replaced = re.sub('n', ' ', a_replaced)
    a_replaced = re.sub(' +', ' ', a_replaced)
    return a_replaced


df['Review_Words'] = df['Review'].apply(lambda x: len(x.split()))

df['Review_Date_Cleaned'] = pd.to_datetime(df['Review_Date']).dt.date

df['Total_Syllables'] = df['Review'].apply(lambda x: syllables.estimate(x))
df['Average_Syllables'] = df['Total_Syllables'] / df['Review_Words']

df['flesch_reading_ease'] = df['Review'].apply(lambda x: textstat.flesch_reading_ease(x))
# Example of a negative readability example
a = df.sort_values(by='flesch_reading_ease').head().iloc[1]
print(a['flesch_reading_ease'])
print()
print(a['Review'])

w_tokenizer = tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(clean_text(text.lower())) if
            w not in stopwords.words('english')]


df['review_lemmas'] = df['Review'].apply(lambda x: lemmatize_text(x))

pd.to_datetime(df['Review_Date']).dt.year.value_counts()


df['Period'] = np.where(pd.to_datetime(df['Review_Date']).dt.year>=2012,'c. Post 2011','Other')
df['Period'] = np.where(pd.to_datetime(df['Review_Date']).dt.year<2012,'b. Btn 2002 and 2011',df['Period'])
df['Period'] = np.where(pd.to_datetime(df['Review_Date']).dt.year<2002,'a. During 2001',df['Period'])
df['Period'].value_counts()

print(df['Rating'].isnull().sum())
print(df['Rating'].agg(['mean','median','count']).round(2))
df['Rating'].value_counts().reset_index().sort_values(by='index').plot.barh(x='index', y='Rating', color="purple")