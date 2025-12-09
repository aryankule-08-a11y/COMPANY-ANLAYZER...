# app.py
st.set_page_config(page_title="Company Analyzer", layout="wide")


st.title("Company Analyzer — Professional Dashboard")
st.markdown("Provide a company name and its website URL; the app will fetch, analyze, and visualize public signals.")


with st.sidebar.form(key='inputs'):
company_name = st.text_input('Company name', placeholder='Acme Inc')
website = st.text_input('Company website (full URL)', placeholder='https://www.example.com')
run = st.form_submit_button('Analyze')


if run:
if not company_name or not website:
st.error('Please provide both company name and website URL.')
else:
analyzer = CompanyAnalyzer(company_name.strip(), website.strip())
with st.spinner('Fetching and analyzing...'):
result = analyzer.run_all()


# Top row: basic info
col1, col2, col3 = st.columns([3,2,2])
with col1:
st.subheader(result['title'] or company_name)
st.write(result['meta_description'] or '—')
st.markdown(f"**URL:** {result['url']}")
if result.get('found_social'):
st.markdown('**Social links found:**')
for s in result['found_social']:
st.write(f"- {s}")
with col2:
st.metric('Pages fetched', result.get('pages_fetched', 1))
st.metric('Readability (Flesch)', f"{result.get('flesch_reading_ease', '—')}")
with col3:
st.metric('Estimated sentiment', result.get('sentiment_label', '—'))
st.metric('Top keyword', result['top_keywords'][0] if result['top_keywords'] else '—')


st.markdown('---')


# Main: visuals
st.subheader('Content & Keywords')
k1, k2 = st.columns([2,1])
with k1:
st.plotly_chart(result['keyword_bar'], use_container_width=True)
st.write('Top extracted sentences (company description candidates):')
for s in result['top_sentences'][:5]:
st.write('> ' + s)
with k2:
st.image(result['wordcloud_image'], caption='Wordcloud', use_column_width=True)


st.subheader('Topic decomposition')
st.plotly_chart(result['topic_bar'], use_container_width=True)


st.subheader('Sentiment & Tone')
s1, s2 = st.columns(2)
with s1:
st.plotly_chart(result['sentiment_pie'], use_container_width=True)
with s2:
st.write('Sentiment scores (VADER)')
st.table(result['sentiment_scores'])


st.markdown('---')
st.subheader('Executive Summary (Auto-generated)')
st.markdown(result['executive_summary'])


st.markdown('---')
st.write('**Raw metadata & diagnostics**')
st.json(result['diagnostics'])


st.success('Analysis complete — use the executive summary & visuals for reports.')
