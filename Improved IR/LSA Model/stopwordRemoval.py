from util import *

# Add your import statements here
from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		"""
		Stopword removal using list of stopwords from NLTK

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		stopwords_set = set(stopwords.words('english'))

		for sentence in text:
			filtered_sentence = [token for token in sentence if token.lower() not in stopwords_set]
			stopwordRemovedText.append(filtered_sentence)


		return stopwordRemovedText
	

	def fromCorpus(self,text,docs):
		"""
		Stopword removal using botton-up approach: IDF measure

		Parameters
		----------
		text : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		docs : list
			A list of list of lists where each sublist refers to a document of 
			corpus whose sub-list is a sequence of tokens representing the document sentences

		idf_threshold: float
			Threshold value (default = 1.0) to check stopwords comparing with IDF score.

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		stopwordRemovedText = []

		stopwords_set = set([word.lower() for word in stopWordList(docs)])

		for sentence in text:
			filtered_sentence = [token for token in sentence if token.lower() not in stopwords_set]
			stopwordRemovedText.append(filtered_sentence)


		return stopwordRemovedText



	