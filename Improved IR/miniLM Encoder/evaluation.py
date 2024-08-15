from util import *

# Add your import statements here
import numpy as np
from math import log2


class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		intersection = set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs)

		precision = len(intersection)/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		precisions = []
		for id in range(len(query_ids)):
			# getting ground truths from func ground_truth_ids
			precision = self.queryPrecision(doc_IDs_ordered[id],query_ids[id],ground_truth_ids(query_ids[id],qrels),k)
			precisions.append(precision)

		meanPrecision = np.mean(precisions)
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		intersection = set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs)
		recall = len(intersection)/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		recalls = []
		for id in range(len(query_ids)):
			recall = self.queryRecall(doc_IDs_ordered[id],query_ids[id],ground_truth_ids(query_ids[id],qrels),k)
			recalls.append(recall)
		
		meanRecall = np.mean(recalls)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,k)
		recall = self.queryRecall(query_doc_IDs_ordered,query_id,true_doc_IDs,k)

		if precision == 0 and recall == 0:
			return 0

		fscore = (2*precision*recall) / (precision+recall)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		f1_scores = []
		for id in range(len(query_ids)):
			f1_scores.append(self.queryFscore(doc_IDs_ordered[id],query_ids[id],ground_truth_ids(query_ids[id],qrels),k))

		meanFscore = np.mean(f1_scores)
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value
			
		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		DCGk = 0
		IDCGk = 0
		retrieved_docs = query_doc_IDs_ordered[:k]

		rel_ids = []
		for id in range(1,k+1):
			rel_id = 0
			doc_id = id - 1
			if retrieved_docs[doc_id] in true_doc_IDs:
				rel_id = self.true_rels[true_doc_IDs.index(retrieved_docs[doc_id])]
			DCGk += rel_id / log2(id + 1)
		
		rels_d = self.true_rels.copy()

		if len(rels_d) < k:
			for id in range(len(rels_d),k):
				rels_d.append(0)

		IDCGk = sum([rels_d[id-1] / log2(id + 1) for id in range(1,k+1)])

		if IDCGk==0:
			return 0

		nDCG = DCGk/IDCGk

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		ndcgs = []
		for id,query_id in enumerate(query_ids):
			rel_docs = [(d["position"],int(d["id"])) for d in qrels if d["query_num"] == str(query_id)]
			rel_docs.sort()

			true_doc_ids = [e[1] for e in rel_docs]
			self.true_rels = [5 - e[0] for e in rel_docs]

			ndcgs.append(self.queryNDCG(doc_IDs_ordered[id], query_id, true_doc_ids, k))

		meanNDCG = np.sum(ndcgs)/len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		ids = []
		for i in range(k):
			if query_doc_IDs_ordered[i] in true_doc_IDs:
				ids.append(i)
		if len(ids)==0:
			return 0
		
		precisions = []
		for i in ids:
			precisions.append(self.queryPrecision(query_doc_IDs_ordered,query_id,true_doc_IDs,i+1))

		avgPrecision = np.sum(precisions)/len(precisions)

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		avgPrecisions = []

		for i in range(len(query_ids)):
			avgPrecisions.append(self.queryAveragePrecision(doc_IDs_ordered[i],query_ids[i],ground_truth_ids(query_ids[i],q_rels),k))

		meanAveragePrecision = np.sum(avgPrecisions)/len(avgPrecisions)
		return meanAveragePrecision

