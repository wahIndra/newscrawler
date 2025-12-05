from GoogleNews import GoogleNews
import datetime

def search_news(query, limit=10, start_date=None, end_date=None, lang='id', region='ID'):
    """
    Search for news articles using GoogleNews.
    
    Args:
        query (str): The search query.
        limit (int): Number of results to return.
        start_date (str, optional): Start date in 'MM/DD/YYYY' format.
        end_date (str, optional): End date in 'MM/DD/YYYY' format.
        lang (str): Language code (default: 'id').
        region (str): Region code (default: 'ID').
        
    Returns:
        list: A list of result dictionaries containing 'title', 'link', 'date', etc.
    """
    print(f"Searching Google News for: {query} (Limit: {limit}, Date: {start_date}-{end_date})")
    
    googlenews = GoogleNews(lang=lang, region=region)
    
    if start_date and end_date:
        googlenews.set_time_range(start_date, end_date)
        
    googlenews.search(query)
    
    # Fetch results. GoogleNews fetches 10 per page.
    # We might need to fetch multiple pages to reach the limit.
    results = googlenews.result()
    
    # If we need more than 10, we can try to get more pages
    # Note: GoogleNews library page fetching can be tricky, but let's try basic pagination
    page = 2
    while len(results) < limit:
        googlenews.get_page(page)
        new_results = googlenews.result()
        if not new_results or len(new_results) == len(results):
            break
        results = new_results
        page += 1
        
    # Deduplicate and limit
    unique_results = []
    seen_links = set()
    
    for item in results:
        link = item.get('link')
        if link and link not in seen_links:
            seen_links.add(link)
            unique_results.append(item)
            if len(unique_results) >= limit:
                break
                
    # Return list of links for compatibility with existing app logic, 
    # OR return full objects if we want to use dates later.
    # The plan said "Ensure it returns publication dates along with URLs".
    # So let's return the full item dictionaries.
    # The app.py will need to be updated to handle this list of dicts instead of list of strings.
    
    return unique_results

if __name__ == "__main__":
    # Test
    results = search_news("Presiden Prabowo", limit=5)
    for res in results:
        print(f"[{res.get('date')}] {res.get('title')} - {res.get('link')}")

