from util import *

# Add your import statements here
from nltk.stem import PorterStemmer



class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		stemmer = PorterStemmer()

		for sentence in text:
			reducedText.append([stemmer.stem(token) for token in sentence])
		
		return reducedText


