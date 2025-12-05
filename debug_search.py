from googlesearch import search

try:
    print("Start search...")
    results = search("test", num_results=5, lang="en")
    print(f"Results type: {type(results)}")
    for i, res in enumerate(results):
        print(f"{i}: {res}")
    print("End search.")
except Exception as e:
    print(f"Error: {e}")
