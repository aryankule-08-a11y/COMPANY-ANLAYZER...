
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
        return list(names[idx]), scores[idx]

    def get_topics(self, doc, n_topics=4, n_top=6):
        vect = TfidfVectorizer(stop_words='english', max_features=2000)
        X = vect.fit_transform([doc])
        if X.shape[1] < 5:
            return []
        svd = TruncatedSVD(n_components=min(n_topics, X.shape[1]-1), random_state=0)
        svd.fit(X)
        terms = vect.get_feature_names_out()
        topics = []
        for comp in svd.components_:
            terms_idx = np.argsort(comp)[::-1][:n_top]
            topics.append([terms[i] for i in terms_idx])
        return topics

    def get_sentiment(self, doc):
        sia = SentimentIntensityAnalyzer()
        sentences = nltk.tokenize.sent_tokenize(doc)
        scores = [sia.polarity_scores(s) for s in sentences]
        df = pd.DataFrame(scores)
        agg = df.mean().to_dict()
        # label
        if agg['compound'] >= 0.05:
            label = 'Positive'
        elif agg['compound'] <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'
        return agg, label, sentences

    def make_wordcloud_image(self, doc, width=600, height=400):
        wc = WordCloud(width=width, height=height, background_color='white').generate(doc)
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        return buf

    def run_all(self):
        diagnostics = {}
        html = self.fetch(self.url)
        diagnostics['fetched_url'] = self.url
        text = self.extract_text(html)
        title, desc, social = self.extract_meta(html)
        diagnostics['found_social'] = social

        # fallback description
        best_desc = desc or (text.split('.')[:2] and '.'.join(text.split('.')[:2]))

        # Keywords
        top_keywords, kw_scores = self.get_top_keywords(text, n=20)
        keyword_bar = None
        if top_keywords:
            df_k = pd.DataFrame({'keyword': top_keywords, 'score': kw_scores})
            fig = go.Figure([go.Bar(x=df_k['keyword'], y=df_k['score'])])
            fig.update_layout(title='Top keywords (TF-IDF)', xaxis_tickangle=-45, height=400)
            keyword_bar = fig

        # Topics
        topics = self.get_topics(text, n_topics=4)
        topic_bar = None
        if topics:
            topic_counts = {f'Topic {i+1}': len(t) for i,t in enumerate(topics)}
            fig2 = go.Figure([go.Bar(x=list(topic_counts.keys()), y=list(topic_counts.values()))])
            fig2.update_layout(title='Topics found', height=300)
            topic_bar = fig2

        # Sentiment
        sentiment_scores, sentiment_label, sentences = self.get_sentiment(text)
        sentiment_pie = go.Figure(data=[go.Pie(labels=['pos','neu','neg'], values=[sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg']])])
        sentiment_pie.update_layout(title='Sentiment distribution (average over sentences)')

        # Wordcloud
        wc_img = self.make_wordcloud_image(text)

        # Readability
        flesch = None
        try:
            flesch = round(textstat.flesch_reading_ease(text), 1)
        except Exception:
            flesch = None

        # top sentences for description
        top_sentences = sorted(sentences, key=lambda s: len(s))[:8] if sentences else []

        # Executive summary (simple heuristic)
        exec_summary = []
        exec_summary.append(f"**Profile:** {title or self.name} — {best_desc}")
        exec_summary.append(f"**Top keywords:** {', '.join(top_keywords[:8])}")
        exec_summary.append(f"**Tone:** {sentiment_label} (compound={sentiment_scores.get('compound'):.2f})")
        if flesch is not None:
            exec_summary.append(f"**Readability (Flesch Reading Ease):** {flesch}")
        if social:
            exec_summary.append(f"**Social links detected:** {', '.join(social[:5])}")
        exec_summary.append('\n**Suggestions:**')
        exec_summary.append('- For investor-facing materials: simplify long paragraphs (target Flesch > 50) and emphasize 3 product/service bullets.')
        exec_summary.append('- Improve sentiment balance if negative signals detected: add customer success stories and case studies.')

        return {
            'title': title,
            'meta_description': desc,
            'url': self.url,
            'found_social': social,
            'pages_fetched': 1,
            'flesch_reading_ease': flesch,
            'top_keywords': top_keywords,
            'keyword_bar': keyword_bar,
            'topic_bar': topic_bar,
            'sentiment_pie': sentiment_pie,
            'sentiment_scores': pd.DataFrame([sentiment_scores]).T.rename(columns={0:'score'}) if sentiment_scores else {},
            'sentiment_label': sentiment_label,
            'wordcloud_image': wc_img,
            'top_sentences': top_sentences,
            'executive_summary': '\n\n'.join(exec_summary),
            'diagnostics': diagnostics
        }
```

---

### 4) `README.md` (usage + Git & GitHub instructions)

````markdown
# Company Analyzer

A Streamlit app to analyze a company's public website. Input the company name + website and get a professional dashboard with NLP, sentiment, keywords, topics, and an executive summary.

## Run locally

1. Create a virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Deploy to Streamlit Cloud or Vercel

- Push this repo to GitHub.
- Use Streamlit Cloud: connect your repo and deploy the `app.py` file as the app entrypoint.

## GitHub setup (one-liners)

```bash
git init
git add .
git commit -m "Initial Company Analyzer"
# create a new GitHub repo via the website and then
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
````

## Notes & next steps

* Add caching (Streamlit `@st.cache_data`) to avoid repeated fetches.
* Add optional OAuth to query paid data sources (Crunchbase, Clearbit) for richer firmographics when you have API keys.
* Add PDF export of the executive summary (report) and an option to download visuals.

```

---

## Prompt to create this app (for use with ChatGPT / Copilot)

```

You are an expert Python developer tasked with building a professional Streamlit web app called "Company Analyzer". The app receives only a company name and a company website URL, fetches the public website, and produces a polished analytics dashboard that includes:

* Site metadata (title, description, social links)
* Extracted textual content
* Top keywords (TF-IDF) and a keyword bar chart
* Topic decomposition (SVD/NMF) displayed as a chart
* Sentence-level sentiment analysis using VADER, and a sentiment breakdown chart
* Readability score (Flesch Reading Ease)
* Wordcloud image of the site's text
* Auto-generated executive summary and suggestions

Build with Streamlit and Python using best practices: modular code (utils.py), requirements.txt, clear UI layout, responsive charts with Plotly, graceful error handling, and caching for repeated runs. Also include a README with instructions for local run and deployment to Streamlit Cloud, and a short checklist of improvements (adding paid data sources, PDF export, caching, GitHub actions).

Deliverables: a fully working `app.py`, `utils.py`, `requirements.txt`, and `README.md` in a GitHub-ready layout. Keep external dependencies minimal and document any API key needs for optional enrichments.

```

---

## Final notes

- This repo is intentionally minimal and designed to be extended: connect Clearbit/Crunchbase/LinkedIn APIs for firmographics, or plug in an LLM (OpenAI) to generate richer executive summaries (note: will require API keys).
- If you'd like, I can convert this into a ready-to-run GitHub repository (zipped) with a GitHub Actions workflow to run tests.

Good luck — want me to add automatic GitHub creation scripts or an LLM-powered executive summary next?

```
