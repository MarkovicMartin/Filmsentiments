import collections
import itertools
import re
import time
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import syllables
import textstat

from nltk import ngrams, tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scrapy.selector import Selector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from wordcloud import STOPWORDS, WordCloud

warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()
ROOT.withdraw()
# the input dialog
USER_INP = simpledialog.askstring(title="Film name input", prompt="Give me adress of reviews you want to analyse")



def download_reviews (url):
    driver = webdriver.Chrome('chromedriver.exe')
    #url = 'https://www.imdb.com/title/tt0241527/reviews?ref_=tt_sa_3' testovaci odkaz na HP film 
    time.sleep(1)
    driver.get(url)



    film_title = driver.title.split('-')[0]

    title_name = driver.title.split(')')
    time.sleep(1)
    body = driver.find_element(By.CSS_SELECTOR, 'body')
    release_date = re.findall(r'\b\d+\b', driver.find_element(By.CSS_SELECTOR, "#main > section > div.subpage_title_block > div > div > h3").text)


    sel = Selector(text=driver.page_source)
    review_counts = sel.css('.lister .header span::text').extract_first().replace(',', '').split(' ')[0].replace(u"\xa0","")

    more_review_pages = round((int(review_counts) / 25) + 0.5)

    for i in tqdm(range(more_review_pages)):
        page_count = 0
        try:
            css_selector = 'load-more-trigger'
            #time.sleep(2)
            # driver.find_element(By.ID, css_selector).click()
            driver_wait = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, 'load-more-trigger')))
            driver.find_element(By.ID, css_selector).click()
            page_count += 1
        except:
            pass

    reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')
    first_review = reviews[0]
    sel2 = Selector(text=first_review.get_attribute('innerHTML'))
    rating = sel2.css('.rating-other-user-rating span::text').extract_first().strip()

    rating_list = []
    review_date_list = []
    review_title_list = []
    author_list = []
    review_list = []
    review_url_list = []
    error_url_list = []
    error_msg_list = []
    reviews = driver.find_elements(By.CSS_SELECTOR, 'div.review-container')

    for d in tqdm(reviews):
        try:
            sel2 = Selector(text=d.get_attribute('innerHTML'))
            try:
                rating = sel2.css('.rating-other-user-rating span::text').extract_first()
            except:
                rating = np.NaN
            try:
                review = sel2.css('.text.show-more__control::text').extract_first()
            except:
                review = np.NaN
            try:
                review_date = sel2.css('.review-date::text').extract_first()
            except:
                review_date = np.NaN
            try:
                author = sel2.css('.display-name-link a::text').extract_first()
            except:
                author = np.NaN
            try:
                review_title = sel2.css('a.title::text').extract_first()
            except:
                review_title = np.NaN
            try:
                review_url = sel2.css('a.title::attr(href)').extract_first()
            except:
                review_url = np.NaN
            rating_list.append(rating)
            review_date_list.append(review_date)
            review_title_list.append(review_title)
            author_list.append(author)
            review_list.append(review)
            review_url_list.append(review_url)
        except Exception as e:
            error_url_list.append(url)
            error_msg_list.append(e)
    review_df = pd.DataFrame({
        'Review_Date': review_date_list,
        'Author': author_list,
        'Rating': rating_list,
        'Review_Title': review_title_list,
        'Review': review_list,
        'Review_Url': review_url,
    })


    review_df.to_csv(f'{film_title}reviews.csv', index=False)
    return review_df

df = download_reviews(USER_INP)
df['Review'].apply(lambda x : clean_text(x))

def create_period(a, release_year):
    current_year = datetime.datetime.now().year
    years_since_release = current_year - release_year
    period_1_end = release_year + (years_since_release // 3)
    period_2_end = release_year + (2 * (years_since_release // 3))
    a['Period'] = np.where(pd.to_datetime(a['Review_Date']).dt.year >= period_2_end, 'c. Post ' + str(period_2_end), 'Other')
    a['Period'] = np.where(pd.to_datetime(a['Review_Date']).dt.year < period_2_end, 'b. Btw ' + str(period_1_end) + ' and ' + str(period_2_end), a['Period'])
    a['Period'] = np.where(pd.to_datetime(a['Review_Date']).dt.year < period_1_end, 'a. During ' + str(release_year), a['Period'])
    a['Period'].value_counts()

def clean_text(a):
    a_replaced = re.sub('[^A-Za-z0-9]+', ' ', a)
    a_replaced = re.sub(r'w+:/{2}[dw-]+(.[dw-]+)*(?:(?:/[^s/]*))*', '', a_replaced)
    a_replaced = re.sub(' +', ' ', a_replaced)    
    return a_replaced

# Working with reviews text
def show_syllables(a):
    a['Review_Words'] = a['Review'].apply(lambda x : len( x.split()) )

    #Counting syllables 
    a['Total_Syllables'] = a['Review'].apply(lambda x : syllables.estimate(x))
    a['Average_Syllables'] = a['Total_Syllables']/a['Review'].apply(lambda x : len( x.split())) 
    return a
if st.button('show syllables'):
    show_syllables(review_df)

def reading_ease(a):
    a['flesch_reading_ease'] = a['Review'].apply(lambda x : textstat.flesch_reading_ease(x))

#display(df.groupby('Rating')['flesch_reading_ease'].agg(['mean','median','count']).round(2))
#df.groupby('Period')['flesch_reading_ease'].agg(['mean','median','count']).round(2)

"""
# We can show worst /best scored reviews

worst_flesh_score = df.sort_values(by='flesch_reading_ease').iloc[0]
best_flesh_score = df.sort_values(by='flesch_reading_ease').iloc[-1]
print('Flesh reading ease score worst: ', second_worst['flesch_reading_ease'], '\n \n Review: \n', second_worst['Review'], '\n')
print('Flesh reading ease score best: ', best['flesch_reading_ease'], '\n \n Review: \n', best['Review'], '\n')
"""

def show_null_reviews_and_rating (a):
    print('Number of Null score reviews',a['Rating'].isnull().sum(), '\n')
    print(a['Rating'].agg(['mean','median','count']).round(2))
    a['Rating'].value_counts().reset_index().sort_values(by='index').plot.barh(x='index', y='Rating', color="blue")

#Working with Date
def date_to_numeric(a):
    a['Review_Date_numeric'] = pd.to_datetime(a['Review_Date']).dt.date

def day_of_review_graph(a):
    a['Review_Day'] = pd.to_datetime(a['Review_Date']).dt.day_name()
    a['Review_Day_no'] = pd.to_datetime(a['Review_Date']).dt.dayofweek
    
    c = df.groupby(['Review_Day_no','Review_Day']).Review.count().reset_index().sort_values(['Review_Day_no'],ascending=False)
    c.plot.barh(x='Review_Day', y='Review', color="blue")

def text_lemmatize (a):
    w_tokenizer = tokenize.WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()
    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(clean_text(text.lower())) if w not in stopwords.words('english')]
    a['review_lemmas'] = a['Review'].apply(lambda x : lemmatize_text(x))

# OTHER analysis
def start_letter_review (a, release_year):
    if 'review_lemmas' not in a.columns:
        text_lemmatize (a)
    if 'Period' not in a.columns:
        create_period(a,release_year)

    full_start_letter_df = pd.DataFrame()
    for period in sorted(a['Period'].unique()):
        curr_lemmatized_tokens = list(a[a['Period']==period]['review_lemmas'])
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
def start_letter_review (a, release_year, periods = False):
    if 'review_lemmas' not in a.columns:
        text_lemmatize (a)
    if 'Period' not in a.columns:
        create_period(a,release_year)
    if period == False:
        lemmatized_tokens = list(a['review_lemmas'])
        #%matplotlib inline
        token_list = list(itertools.chain(*lemmatized_tokens)) 
        counts_no = collections.Counter(token_list) 
        # counts_no = collections.Counter(ngrams(token_list, 1))
        clean_reviews = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count']) 
        fig, ax = plt.subplots(figsize=(12, 8)) 
        clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
        ax.set_title("Most Frequently used words in Reviews") 
        plt.show()
    else:
        for period in sorted(df['Period'].unique()):
            lemmatized_tokens = list(df[df['Period']==period]['review_lemmas'])
            token_list = list(itertools.chain(*lemmatized_tokens)) 
            counts_no = collections.Counter(token_list) 
            clean_reviews = pd.DataFrame(counts_no.most_common(10), columns=['words', 'count']) 
            fig, ax = plt.subplots(figsize=(12, 4)) 
            clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
            ax.set_title("Most Frequently used words in Reviews Period( "+str(period)+")") 
            plt.show()

def grams(a,how_many, release_year,period = False):
#Bigrams
    if 'review_lemmas' not in a.columns:
        text_lemmatize (a)
    if 'Period' not in a.columns:
        create_period(a,release_year)
    if period == False:
        lemmatized_tokens = list(a[a['Period']==period]['review_lemmas'])
        token_list = list(itertools.chain(*lemmatized_tokens)) 
        counts_no = collections.Counter(ngrams(token_list, how_many))
        clean_reviews = pd.DataFrame(counts_no.most_common(30), columns=['words', 'count']) 
        fig, ax = plt.subplots(figsize=(12, 8)) 
        clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
        ax.set_title("Most Frequently used Bigrams in Reviews") 
        plt.show()

    else:
        for period in sorted(a['Period'].unique()):
            lemmatized_tokens = list(df[df['Period']==period]['review_lemmas'])
            token_list = list(itertools.chain(*lemmatized_tokens)) 
            counts_no = collections.Counter(token_list) 
            clean_reviews = pd.DataFrame(counts_no.most_common(10), columns=['words', 'count']) 
            fig, ax = plt.subplots(figsize=(12, 4)) 
            clean_reviews.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="purple") 
            ax.set_title("Most Frequently used words in Reviews Period( "+str(period)+")") 
            plt.show()

def word_cloud (a):
    if 'review_lemmas' not in a.columns:
        text_lemmatize (a)
#Word Cloud
    for rating in range(1,11):
        curr_lemmatized_tokens = list(a[a['Rating']==rating]['review_lemmas'])
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
