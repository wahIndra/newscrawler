from crawler.searcher import search_news
from crawler.scraper import get_article_content
from model.sentiment import SentimentAnalyzer
import sys
import datetime

def verify():
    print("1. Testing Search (GoogleNews)...")
    # Test 1: Basic Search
    print("   - Basic Search 'Indonesia'...")
    results = search_news("Indonesia", limit=2)
    if not results:
        print("FAIL: No results found.")
        return
    print(f"PASS: Found {len(results)} results.")

    print("\n2. Testing Scraper...")
    url = results[0]['link']
    print(f"   - Scraping {url}...")
    article = get_article_content(url)
    if article and article.get('text'):
        print(f"PASS: Scraped article title: {article['title']}")
    else:
        print("FAIL: Could not scrape article.")
        return

    print("\n3. Testing Model (Ensemble & Advanced)...")
    try:
        analyzer = SentimentAnalyzer()
        text = "Ekonomi Indonesia sedang tumbuh pesat. Bisnis cuan gede dan mantul banget!"
        print(f"   - Analyzing text: '{text}'")
        
        # Test Ensemble
        print("   - Testing Ensemble (IndoBERT + XLM-RoBERTa)...")
        result = analyzer.predict(text, model_names=["IndoBERT (ID)", "XLM-RoBERTa (Multilingual)"])
        print(f"PASS: Result: {result}")
        
        # Test Explanation
        print("   - Testing Explanation...")
        explanation = analyzer.explain_sentiment(result['label'], result['score'], text)
        print(f"PASS: Explanation: {explanation}")
        
        # Test Markers
        print("   - Testing Markers...")
        markers = analyzer.get_sentiment_markers(text)
        print(f"PASS: Markers: {markers}")
        
    except Exception as e:
        print(f"FAIL: Model error: {e}")

if __name__ == "__main__":
    verify()
