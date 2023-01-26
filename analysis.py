#IMPORTS
import pandas as pd
import numpy as np
import syllables
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import tokenize,ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import textstat
import itertools
import collections
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS


def clean_text(a):
    a_replaced = re.sub('[^A-Za-z0-9]+', ' ', a)
    a_replaced = re.sub(r'w+:/{2}[dw-]+(.[dw-]+)*(?:(?:/[^s/]*))*', '', a_replaced)
    a_replaced = re.sub(' +', ' ', a_replaced)    
    return a_replaced


#INPUT DATA we downloaded in imdb script
df = pd.read_csv('out.csv')

#released_date =  


# Working with reviews text
df['Review_Words'] = df['Review'].apply(lambda x : len( x.split()) )

#Counting syllables 
df['Total_Syllables'] = df['Review'].apply(lambda x : syllables.estimate(x))
df['Average_Syllables'] = df['Total_Syllables']/df['Review_Words']

df['flesch_reading_ease'] = df['Review'].apply(lambda x : textstat.flesch_reading_ease(x))

#display(df.groupby('Rating')['flesch_reading_ease'].agg(['mean','median','count']).round(2))
#df.groupby('Period')['flesch_reading_ease'].agg(['mean','median','count']).round(2)

"""
# We can show worst /best scored reviews

worst_flesh_score = df.sort_values(by='flesch_reading_ease').iloc[0]
best_flesh_score = df.sort_values(by='flesch_reading_ease').iloc[-1]
print('Flesh reading ease score worst: ', second_worst['flesch_reading_ease'], '\n \n Review: \n', second_worst['Review'], '\n')
print('Flesh reading ease score best: ', best['flesch_reading_ease'], '\n \n Review: \n', best['Review'], '\n')
"""
w_tokenizer = tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(clean_text(text.lower())) if w not in stopwords.words('english')]

df['review_lemmas'] = df['Review'].apply(lambda x : lemmatize_text(x))

print('Number of Null score reviews',df['Rating'].isnull().sum(), '\n')
print(df['Rating'].agg(['mean','median','count']).round(2))
df['Rating'].value_counts().reset_index().sort_values(by='index').plot.barh(x='index', y='Rating', color="blue")



#Working with Date
df['Review_Date_Cleaned'] = pd.to_datetime(df['Review_Date']).dt.date

df['Review_Day'] = pd.to_datetime(df['Review_Date']).dt.day_name()
df['Review_Day_no'] = pd.to_datetime(df['Review_Date']).dt.dayofweek
c = df.groupby(['Review_Day_no','Review_Day']).Review.count().reset_index().sort_values(['Review_Day_no'],ascending=False)
c.plot.barh(x='Review_Day', y='Review', color="blue")

#Make this relevant for chosen film, not for preprepared Harry potter  (released in 2001)
df['Period'] = np.where(pd.to_datetime(df['Review_Date']).dt.year>=2012,'c. Post 2011','Other')
df['Period'] = np.where(pd.to_datetime(df['Review_Date']).dt.year<2012,'b. Btw 2002 and 2011',df['Period'])
df['Period'] = np.where(pd.to_datetime(df['Review_Date']).dt.year<2002,'a. During 2001',df['Period'])
df['Period'].value_counts()

# OTHER analysis
full_start_letter_df = pd.DataFrame()
for period in sorted(df['Period'].unique()):
    curr_lemmatized_tokens = list(df[df['Period']==period]['review_lemmas'])
    curr_token_list = list(itertools.chain(*curr_lemmatized_tokens))
    start_letter = [i[0] for i in curr_token_list]
    start_letter_df = (pd.DataFrame(start_letter)[0].value_counts(1)*100).reset_index().sort_values(by='index')
    start_letter_df[0] = start_letter_df[0]
    start_letter_df.columns = ['letter',period]
    start_letter_df['Start_Letter'] = np.where(start_letter_df['letter'].isin(['a','e','i','o','u']),'a. Vowel',
                                                  np.where(start_letter_df['letter'].isin(['0','1','2','3','4','5','6','7','8','9']),'c. Number',
                                                       'b. Consonant')
                                              )
    start_letter_df = start_letter_df.groupby('Start_Letter')[period].sum().reset_index()
    start_letter_df.columns = ['Start_Letter',period]
    start_letter_df[period] = start_letter_df[period].apply(lambda x : np.round(x,2))
    try:
        full_start_letter_df = full_start_letter_df.merge(start_letter_df)
    except:
        full_start_letter_df = start_letter_df
print(full_start_letter_df.shape)
full_start_letter_df


#Graphs and visualization

lemmatized_tokens = list(df['review_lemmas'])
#%matplotlib inline
token_list = list(itertools.chain(*lemmatized_tokens)) 
counts_no = collections.Counter(token_list) 
# counts_no = collections.Counter(ngrams(token_list, 1))
clean_reviews = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count']) 
fig, ax = plt.subplots(figsize=(12, 8)) 
clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
ax.set_title("Most Frequently used words in Reviews") 
plt.show()


for period in sorted(df['Period'].unique()):
    lemmatized_tokens = list(df[df['Period']==period]['review_lemmas'])
    token_list = list(itertools.chain(*lemmatized_tokens)) 
    counts_no = collections.Counter(token_list) 
    clean_reviews = pd.DataFrame(counts_no.most_common(10), columns=['words', 'count']) 
    fig, ax = plt.subplots(figsize=(12, 4)) 
    clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
    ax.set_title("Most Frequently used words in Reviews Period( "+str(period)+")") 
    plt.show()

#Bigrams
counts_no = collections.Counter(ngrams(token_list, 2))
clean_reviews = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count']) 
fig, ax = plt.subplots(figsize=(12, 8)) 
clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
ax.set_title("Most Frequently used Bigrams in Reviews") 
plt.show()

counts_no = collections.Counter(ngrams(token_list, 2))
clean_reviews = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count']) 
fig, ax = plt.subplots(figsize=(12, 8)) 
clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
ax.set_title("Most Frequently used Bigrams in Reviews") 
plt.show()

#Trigrams
counts_no = collections.Counter(ngrams(token_list, 3))
clean_reviews = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count']) 
fig, ax = plt.subplots(figsize=(12, 8)) 
clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
ax.set_title("Most Frequently used Trigrams in Reviews") 
plt.show()

for period in sorted(df['Period'].unique()):
    lemmatized_tokens = list(df[df['Period']==period]['review_lemmas'])
    token_list = list(itertools.chain(*lemmatized_tokens)) 
    counts_no = collections.Counter(ngrams(token_list, 3))
    clean_reviews = pd.DataFrame(counts_no.most_common(10), columns=['words', 'count']) 
    fig, ax = plt.subplots(figsize=(12, 4)) 
    clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
    ax.set_title("Most Frequently used tri-grams in Reviews Period( "+str(period)+")") 
    plt.show()


#Word Cloud
for rating in range(1,11):
    curr_lemmatized_tokens = list(df[df['Rating']==rating]['review_lemmas'])
    vectorizer = CountVectorizer(ngram_range=(2,2))
    bag_of_words = vectorizer.fit_transform(df[df['Rating']==rating]['review_lemmas'].apply(lambda x : ' '.join(x)))
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    words_dict = dict(words_freq)
    WC_height = 1000
    WC_width = 1500
    WC_max_words = 200
    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)
    wordCloud.generate_from_frequencies(words_dict)
    plt.figure(figsize=(20,8))
    plt.imshow(wordCloud)
    plt.title('Word Cloud for Rating '+str(rating))
    plt.axis("off")
    plt.show()