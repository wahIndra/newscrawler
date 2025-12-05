import streamlit as st
import pandas as pd
import numpy as np
from crawler.searcher import search_news
from crawler.scraper import get_article_content
from model.sentiment import SentimentAnalyzer
import datetime
from dateparser import parse as parse_date

# Page Config
st.set_page_config(
    page_title="Indonesian News Sentiment Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    h1, h2, h3 { color: #2c3e50; }
    @media (prefers-color-scheme: dark) {
        .main-header, h1, h2, h3 { color: #ecf0f1; }
    }
    .marker-pos { background-color: #d4edda; color: #155724; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
    .marker-neg { background-color: #f8d7da; color: #721c24; padding: 2px 5px; border-radius: 3px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize Model (Cached)
@st.cache_resource
def load_model():
    return SentimentAnalyzer()

try:
    with st.spinner("Loading Sentiment Models..."):
        analyzer = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=100)
    st.title("Settings")
    
    query = st.text_input("Subject / Topic", "Presiden Prabowo", help="Enter a person, company, or topic to analyze.")
    
    # Language Selection
    lang_mode = st.radio("Language", ["Bahasa Indonesia (ID)", "English (EN)", "Both (ID + EN)"])
    
    # Model Selection
    st.subheader("🤖 Model Selection")
    available_models = list(SentimentAnalyzer.MODELS.keys())
    default_models = ["IndoBERT (ID)"]
    if "English" in lang_mode or "Both" in lang_mode:
        default_models.append("DistilBERT (EN)")
        
    selected_models = st.multiselect("Choose Models (Ensemble)", available_models, default=default_models[:1])
    if not selected_models:
        st.warning("Please select at least one model.")
    
    st.divider()
    
    search_mode = st.radio("Search Mode", ["Article Count", "Date Range"])
    
    start_date = None
    end_date = None
    num_articles = 10
    
    if search_mode == "Article Count":
        num_articles = st.slider("Number of Articles", min_value=1, max_value=100, value=10, help="Limit the number of articles to crawl.")
    else:
        today = datetime.date.today()
        last_week = today - datetime.timedelta(days=7)
        date_range = st.date_input("Select Date Range", (last_week, today))
        if len(date_range) == 2:
            start_date = date_range[0].strftime("%m/%d/%Y")
            end_date = date_range[1].strftime("%m/%d/%Y")
        else:
            st.warning("Please select both start and end dates.")
    
    st.divider()
    analyze_btn = st.button("Analyze Sentiment", type="primary")
    
    st.divider()
    st.markdown("### About")
    st.info("Advanced Sentiment Analyzer v2.0\nSupports Multi-language, Ensemble Models, and Trend Forecasting.")

# Main Content
st.markdown("<h1 class='main-header'>📰 Advanced News Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>AI-Powered Analysis with Multi-Model Ensemble & Forecasting</p>", unsafe_allow_html=True)
st.divider()

if analyze_btn and query and selected_models:
    # 1. Search
    with st.status("🔍 Searching for news...", expanded=True) as status:
        results_data = []
        
        # Determine languages to search
        langs_to_search = []
        if "ID" in lang_mode or "Both" in lang_mode: langs_to_search.append('id')
        if "EN" in lang_mode or "Both" in lang_mode: langs_to_search.append('en')
        
        for lang in langs_to_search:
            st.write(f"Searching Google News ({lang.upper()}) for **'{query}'**...")
            try:
                # Adjust limit if searching both to avoid double the articles
                limit = num_articles // len(langs_to_search) if search_mode == "Article Count" else num_articles
                res = search_news(query, limit=limit, start_date=start_date, end_date=end_date, lang=lang, region=lang.upper())
                results_data.extend(res)
            except Exception as e:
                st.error(f"Error searching {lang}: {e}")

        st.write(f"Found {len(results_data)} articles total.")
        
        if not results_data:
            status.update(label="No articles found", state="error")
            st.warning("No articles found. Try a different keyword or date range.")
            st.stop()

        analyzed_results = []
        progress_bar = st.progress(0)
        
        # 2. Crawl & Analyze (Multithreaded)
        import concurrent.futures
        
        # Track errors
        errors = []

        def process_single_article(item, analyzer, selected_models):
            url = item.get('link')
            try:
                pub_date = item.get('date')
                
                # Scrape
                article = get_article_content(url)
                if not article or not article.get('text'):
                    return {"error": f"Scraping failed or empty content for {url}"}
                
                # Analyze with Ensemble
                sentiment_result = analyzer.predict(article['text'], model_names=selected_models)
                
                # Generate Explanation
                explanation = analyzer.explain_sentiment(sentiment_result['label'], sentiment_result['score'], article['text'])
                
                # Get Markers
                markers = analyzer.get_sentiment_markers(article['text'])
                
                # Parse date
                parsed_date = None
                if pub_date:
                    parsed_date = parse_date(pub_date)
                if not parsed_date:
                        parsed_date = datetime.datetime.now()

                return {
                    "Title": article['title'],
                    "Sentiment": sentiment_result['label'],
                    "Score": sentiment_result['score'],
                    "URL": url,
                    "Text": article['text'],
                    "Date": parsed_date,
                    "Explanation": explanation,
                    "Markers": markers,
                    "Details": sentiment_result['details']
                }
            except Exception as e:
                return {"error": f"Error processing {url}: {str(e)}"}

        # Reduce max_workers for Cloud stability (1GB RAM limit)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(process_single_article, item, analyzer, selected_models): item for item in results_data}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
                status.update(label=f"Processing article {i+1}/{len(results_data)}...", state="running")
                result = future.result()
                if result:
                    if "error" in result:
                        errors.append(result["error"])
                    else:
                        analyzed_results.append(result)
                progress_bar.progress((i + 1) / len(results_data))
            
        status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        if errors and not analyzed_results:
            st.error("All articles failed to process. Common reasons:")
            st.markdown("- **Scraping Blocked**: News sites often block cloud server IPs.")
            st.markdown("- **Empty Content**: The scraper couldn't extract text.")
            with st.expander("View Error Log"):
                for err in errors:
                    st.write(err)

    # 3. Display Results
    if analyzed_results:
        df = pd.DataFrame(analyzed_results)
        
        # Metrics
        st.subheader("📊 Sentiment Overview")
        col1, col2, col3 = st.columns(3)
        total = len(df)
        positive = len(df[df['Sentiment'] == 'positive'])
        negative = len(df[df['Sentiment'] == 'negative'])
        
        col1.metric("Total Articles", total)
        col2.metric("Positive", f"{positive}", delta=f"{(positive/total)*100:.1f}%")
        col3.metric("Negative", f"{negative}", delta=f"{(negative/total)*100:.1f}%", delta_color="inverse")
        
        # Charts
        st.subheader("📈 Sentiment Analysis & Forecast")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**Sentiment Distribution**")
            chart_data = df['Sentiment'].value_counts().reset_index()
            chart_data.columns = ['Sentiment', 'Count']
            st.bar_chart(chart_data, x='Sentiment', y='Count', color='Sentiment')
            
        with chart_col2:
            st.markdown("**Future Trend (3-Month Forecast)**")
            if not df.empty and 'Date' in df.columns:
                sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                df['NumericSentiment'] = df['Sentiment'].map(sentiment_map)
                df_trend = df.sort_values('Date')
                
                # Linear Regression for Forecast
                if len(df_trend) > 1:
                    df_trend['Days'] = (df_trend['Date'] - df_trend['Date'].min()).dt.days
                    X = df_trend['Days'].values.reshape(-1, 1)
                    y = df_trend['NumericSentiment'].values
                    
                    # Simple Linear Regression
                    from sklearn.linear_model import LinearRegression
                    reg = LinearRegression().fit(X, y)
                    
                    # Forecast 90 days
                    last_day = df_trend['Days'].max()
                    future_days = np.array([last_day + i for i in range(1, 91)]).reshape(-1, 1)
                    future_dates = [df_trend['Date'].max() + datetime.timedelta(days=i) for i in range(1, 91)]
                    future_preds = reg.predict(future_days)
                    
                    # Combine for plotting
                    # We need a unified dataframe for st.line_chart or similar
                    # Let's create a separate forecast dataframe
                    df_forecast = pd.DataFrame({
                        'Date': future_dates,
                        'NumericSentiment': future_preds,
                        'Type': 'Forecast'
                    })
                    df_trend['Type'] = 'Historical'
                    
                    combined_df = pd.concat([df_trend[['Date', 'NumericSentiment', 'Type']], df_forecast])
                    
                    # Streamlit line chart handles simple data best, let's just plot the line
                    # To distinguish, we might need a more complex plotting lib like Altair or Plotly
                    # For now, let's just append the forecast to the line
                    st.line_chart(combined_df, x='Date', y='NumericSentiment')
                    st.caption("Historical data + 3-Month Linear Forecast")
                else:
                    st.line_chart(df_trend, x='Date', y='NumericSentiment')
        
        # Detailed List
        st.subheader("📑 Analyst Insights")
        
        tab1, tab2 = st.tabs(["Detailed Analysis", "Raw Data"])
        
        with tab1:
            for index, row in df.iterrows():
                color = "green" if row['Sentiment'] == 'positive' else "red" if row['Sentiment'] == 'negative' else "gray"
                with st.expander(f":{color}[{row['Sentiment'].upper()}] - {row['Title']}"):
                    st.markdown(f"**Date:** {row['Date'].strftime('%Y-%m-%d')}")
                    st.markdown(f"**Confidence:** `{row['Score']:.4f}`")
                    
                    # Explanation
                    st.info(f"**Analyst Note:** {row['Explanation']}")
                    
                    # Markers
                    markers = row['Markers']
                    if markers['positive'] or markers['negative']:
                        st.markdown("**Key Indicators:**")
                        marker_html = ""
                        for m in markers['positive']:
                            marker_html += f"<span class='marker-pos'>{m}</span> "
                        for m in markers['negative']:
                            marker_html += f"<span class='marker-neg'>{m}</span> "
                        st.markdown(marker_html, unsafe_allow_html=True)
                    
                    st.markdown(f"**Snippet:** _{row['Text'][:300]}..._")
                    st.markdown(f"[Read Full Article]({row['URL']})")
        
        with tab2:
            st.dataframe(df[['Date', 'Title', 'Sentiment', 'Score', 'URL']])

elif analyze_btn and not selected_models:
    st.warning("Please select at least one model to proceed.")
elif analyze_btn and not query:
    st.warning("Please enter a subject to analyze.")
else:
    st.info("👈 Configure your analysis in the sidebar.")



