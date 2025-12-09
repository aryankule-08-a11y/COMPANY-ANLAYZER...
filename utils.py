# utils.py
import requests
from bs4 import BeautifulSoup
import tldextract
import re
import io
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.graph_objects as go
import textstat
from textblob import TextBlob


# Ensure NLTK assets available
nltk.download('vader_lexicon')


class CompanyAnalyzer:
def __init__(self, name, url, user_agent=None):
self.name = name
self.url = url
self.domain = tldextract.extract(url).registered_domain
self.user_agent = user_agent or 'CompanyAnalyzer/1.0 (+https://github.com/)'


def fetch(self, url):
headers = {'User-Agent': self.user_agent}
try:
r = requests.get(url, headers=headers, timeout=12)
r.raise_for_status()
return r.text
except Exception as e:
return ''


def extract_text(self, html):
if not html:
return ''
soup = BeautifulSoup(html, 'lxml')
# remove scripts/styles
for s in soup(['script', 'style', 'noscript']):
s.extract()
text = ' '.join(soup.stripped_strings)
return text


def extract_meta(self, html):
soup = BeautifulSoup(html or '', 'lxml')
title = soup.title.string.strip() if soup.title and soup.title.string else ''
desc = ''
d = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property':'og:description'})
if d and d.get('content'):
desc = d['content'].strip()
# find social links
social = []
for a in soup.find_all('a', href=True):
href = a['href']
if 'twitter.com' in href or 'linkedin.com' in href or 'facebook.com' in href or 'instagram.com' in href:
social.append(href)
return title, desc, social


def get_top_keywords(self, doc, n=15):
if not doc or len(doc.split()) < 30:
return []
vect = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
X = vect.fit_transform([doc])
names = np.array(vect.get_feature_names_out())
scores = np.asarray(X.sum(axis=0)).ravel()
idx = np.argsort(scores)[::-1][:n]
'sentiment_scores': pd.DataFrame([sentiment_scores]).T.rename(c
