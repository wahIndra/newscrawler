import googlesearch
import inspect

print(f"Module file: {googlesearch.__file__}")
print(f"Search signature: {inspect.signature(googlesearch.search)}")
