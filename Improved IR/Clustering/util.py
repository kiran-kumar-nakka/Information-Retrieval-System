# Add your import statements here
import math





# Add any utility functions here

def stopWordList(docs,idf_threshold=1.0):
    """
    Stopword list curated using botton-up approach: IDF measure

    Parameters
    ----------
    arg1 : list
        A list of (docs) list of (setences of docs) lists (tokens of sentence) where each sublist 
        refers to a document of corpus whose sub-list is a sequence of tokens representing the 
        document sentences

    idf_threshold: float
        Threshold value (default = 1.0) to check stopwords comparing with IDF score.

    Returns
    -------
    list
        A list stopwords
    """

    stopword_list = []

    frequency_dict = {}
    total_documents = len(docs)

    for document in docs:
        document_tokens = set(token for sentence in document for token in sentence)
        for token in document_tokens:
            if token in frequency_dict:
                frequency_dict[token] += 1
            else:
                frequency_dict[token] = 1
        
    # idf_scores = []
    for term, freq in frequency_dict.items():
        idf = math.log10(total_documents / freq)
        # idf_scores.append(idf)
        if idf<idf_threshold:
            stopword_list.append(term)
    
    # print(idf_scores)

    return stopword_list


def ground_truth_ids(query_id,qrels):
    ids = []
    for q_dict in qrels:
        if int(q_dict["query_num"]) == int(query_id):
            ids.append(int(q_dict["id"]))
    return ids