from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer



class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# List of punctuation marks to split on
		punctuation_marks = ['.', '!', '?', ';', ':', ',', '"', "'", '(', ')', '[', ']', '{', '}']

		for sentence in text:
			
			# Insert spaces before and after all punctuation marks
			for mark in punctuation_marks:
				# We get punctuation marks as separate tokens
				sentence = sentence.replace(mark, ' ' + mark + ' ')

			# Split the text into tokens based on whitespace
			tokens = sentence.split()

			# Strip extra spaces from tokens
			tokens = [token.strip() for token in tokens]

			tokenizedText.append(tokens)
		

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		t = TreebankWordTokenizer()

		for sentence in text:
			tokenizedText.append(t.tokenize(sentence))

		return tokenizedText