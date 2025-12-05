from newspaper import Article, Config
import nltk
import requests

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_article_content(url):
    """
    Download and parse an article from a URL.
    Handles Google News redirects and adds User-Agent headers.
    """
    # 1. Clean URL (remove Google tracking params if present)
    if "&ved=" in url:
        url = url.split("&ved=")[0]
    
    # 2. Configure Newspaper3k with User-Agent
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10
    
    try:
        # 3. Handle Redirects manually (optional but safer for Google links)
        # Google News links often need to be visited to get the real URL
        if "google.com" in url or "news.google.com" in url:
            try:
                response = requests.get(url, headers={'User-Agent': user_agent}, allow_redirects=True, timeout=10)
                url = response.url
            except Exception as e:
                print(f"Warning: Failed to resolve redirect for {url}: {e}")
                # Continue with original URL if resolve fails
        
        # 4. Download and Parse
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        # Check if text is empty
        if not article.text:
            return None
            
        return {
            "title": article.title,
            "text": article.text,
            "publish_date": article.publish_date,
            "authors": article.authors,
            "url": url
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

if __name__ == "__main__":
    # Test with a known article
    test_url = "https://www.cnnindonesia.com/" 
    print(f"Testing scraper with {test_url}")
    data = get_article_content(test_url)
    if data:
        print(f"Title: {data['title']}")
        print(f"Text length: {len(data['text'])}")
    else:
        print("Failed to scrape.")
