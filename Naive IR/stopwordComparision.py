from nltk.corpus import stopwords
from util import *
import json
import argparse

dataset_path = input("Enter dataset directory path: ")
nltk_stopwords_set = set(stopwords.words('english'))
# Read documents
docs_json = json.load(open(dataset_path + "stopword_removed_docs.txt", 'r'))[:]
docs = [item for item in docs_json]



print(f"Len of docs: {len(docs)}")
manual_stopwords_set = set(stopWordList(docs)) 
print()
print("NLTK stopwords: ")
print(nltk_stopwords_set)
print()
print("IDF stopwords: ")
print(manual_stopwords_set)