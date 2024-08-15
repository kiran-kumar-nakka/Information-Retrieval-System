from util import *

# Add your import statements here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		st = time.time()
		index = None

		#Fill in code here
		self.encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
		corpus = []
		for doc in docs:
			d_doc = []
			for sent in doc:
				for word in sent:
					d_doc.append(word)
			corpus.append(' '.join(d_doc))

		# self.vectorizer = TfidfVectorizer()
		# index = self.vectorizer.fit_transform(corpus)

		#performing LSA
		#index is doc-term matrix
		# X = index.toarray() 
		# self.U,self.S,self.Vt = np.linalg.svd(X,full_matrices=False)

		#term-concept matrix is Vt.transpose
		#doc-concept matrix is U
		#singular values is S 
		# where number of concepts is min(t,d) = 1400 here. --> reducing from 5227 words to 1400 concepts
		
		# self.k = 400
		# self.index = self.U[:,:self.k]@np.diag(self.S[:self.k])

		self.index = self.encoder.encode(corpus)
		self.docIDs = docIDs

		print(f"Built the index in {time.time()-st:4f} seconds")

	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []
		st = time.time()

		#Fill in code here

		queries_flattened = []
		for q in queries:
			dq = []
			for word in q[0]:
				dq.append(word)
			queries_flattened.append(' '.join(dq))
		
		query_index = self.encoder.encode(queries_flattened)
		query_index = np.array(query_index, dtype=np.float64)
		#query_index is query-term matrix
		it = time.time()
		print(f"Query indexing time: {it-st:4f}")
		#transforming the query to new reduced space
		# q_transformed = query_index@self.Vt[:self.k,:].T

		#q_transformed is query-concept matrix

		# similarities = cosine_similarity(q_transformed,self.index)
		similarities = []

		for query in query_index:
			sims = cosine_similarity(query.reshape(1,-1),self.index)[0]
			top_ids = [i for i,sim in sorted(enumerate(sims),key=lambda x:x[1],reverse=True)]
			top_doc_ids = [self.docIDs[doc_id] for doc_id in top_ids]
			doc_IDs_ordered.append(top_doc_ids)

		# for similarity in similarities:
		# 	# we are taking similarities scores of each query
		# 	top_inds = [i for i, _ in sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)]

		# 	top_doc_ids = [self.docIDs[doc_id] for doc_id in top_inds]
		# 	doc_IDs_ordered.append(top_doc_ids)

		print(f"Ranked documents retrieved for queries in {time.time()-it:4f} seconds")
	
		return doc_IDs_ordered




