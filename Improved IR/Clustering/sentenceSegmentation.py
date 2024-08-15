from util import *

# Add your import statements here
from nltk.tokenize import sent_tokenize



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		if not isinstance(text, str):
			raise TypeError("Input must be a string")
		
		sentence_delimiters = ['.','?','!']
		segmentedText = []
		
		current_sentence = ''
		for char in text:
			current_sentence += char
			if char in sentence_delimiters:
				segmentedText.append(current_sentence.strip())
				current_sentence = ''
		
		if current_sentence:
			segmentedText.append(current_sentence.strip())

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""
		
		segmentedText = sent_tokenize(text)
		
		return segmentedText