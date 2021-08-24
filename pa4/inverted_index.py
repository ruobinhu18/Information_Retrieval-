from pathlib import Path
from typing import Union, List, Tuple, Dict
import os
import shelve
from math import sqrt, log2
from nltk import word_tokenize
from utils import timer, load_wapo
from text_processing import TextProcessing
from heapq import heapify

text_processor = TextProcessing.from_nltk()


def get_doc_vec_norm(term_tfs: List[float]) -> float:
    """
    helper function, should be called in build_inverted_index
    compute the length of a document vector
    :param term_tfs: a list of term weights (log tf) for one document
    :return:
    """
    s = 0
    for tf in term_tfs:
        s += tf * tf
    return sqrt(s)


@timer
def build_inverted_index(
        wapo_jl_path: Union[str, os.PathLike],
        index_shelve_path: str,
        doc_vec_norm_shelve_path: str,
) -> None:
    """
    load wapo_pa4.jl to build two shelve files in the provided path

    :param wapo_jl_path:
    :param index_shelve_path: for each normalized term as a key, the value should be a list of tuples;
        each tuple stores the doc id this term appear in and the term weight (log tf)
    :param doc_vec_norm_shelve_path: for each doc id as a key, the value should be the "length" of that document vector
    :return:
    """
    doc_iter = load_wapo(wapo_jl_path)
    dic = {}
    dic_norm = {}
    # create dict
    print('creating the dictionary...')
    while True:
        # get next document
        try:
            doc = next(doc_iter)
        except StopIteration:
            break
        idx = doc['doc_id']
        title = doc['title']
        content = doc['content_str']
        # normalized tokens as a set
        s = text_processor.get_normalized_tokens(title, content)
        s_set = set(s)
        # initialize the term_tfs for each doc_id
        term_tfs = []
        for token in s_set:
            # count the log2(tf)
            tf = text_processor.tf(s.count(token))
            # add the score to the term_tfs list
            term_tfs.append(tf)
            if token not in dic:
                dic[token] = [(idx, tf)]
            else:
                dic[token].append((idx, tf))
        # build the vec_norm dict
        dic_norm[idx] = get_doc_vec_norm(term_tfs)
    # create index shelve
    print('creating the index shelve...')
    batch_size = 2000
    idx = 0
    for key in dic:
        if idx == 0:
            db = shelve.open(index_shelve_path)
        db[key] = dic[key]
        idx += 1
        if idx > batch_size:
            db.close()
            idx = 0
    db.close()

    # create vec_norm shelve
    print('creating the vector norm shelve...')
    idx = 0
    for key in dic_norm:
        if idx == 0:
            db = shelve.open(doc_vec_norm_shelve_path)
        db[str(key)] = dic_norm[key]
        idx += 1
        if idx > batch_size:
            db.close()
            idx = 0
    db.close()


def parse_query(
        query: str, shelve_index: shelve.Shelf
) -> Tuple[List[str], List[str], List[str]]:
    """
    helper function, should be called in query_inverted_index
    given each query, return a list of normalized terms, a list of stop words and a list of unknown words separately

    :param query:
    :param shelve_index:
    :return:
    """
    db = shelve_index
    query_token = sorted(word_tokenize(query))
    stopword = []
    unknown = []
    result = []
    # remove empty string
    query_token = [e for e in query_token if e != '']
    for token in query_token:
        if token in text_processor.STOP_WORDS:
            stopword.append(token)
            continue
        # normalize after comparing with stopwords
        token = text_processor.normalize(token)
        if token not in db:
            unknown.append(token)
        else:
            result.append(token)
    output = (result, stopword, unknown)
    return output


def top_k_docs(doc_scores: Dict[int, float], k: int) -> List[Tuple[float, int]]:
    """
    helper function, should be called in query_inverted_index
    given the doc_scores, return top k doc ids and corresponding scores using a heap
    :param doc_scores: a dictionary where doc id is the key and cosine similarity score is the value
    :param k:
    :return: a list of tuples, each tuple contains (score, doc_id)
    """

    result = []
    num = 0
    for doc_id in doc_scores:
        score = doc_scores[doc_id]
        if num < k:
            result.append((score, doc_id))
            num += 1
        else:
            # replace the minimum score if current score is larger
            if score > result[0][0]:
                result[0] = (score, doc_id)
        heapify(result)
    # sort the result with descent order
    result.sort(reverse=True)
    # add other documents
    idx_lst = [idx for _, idx in result]
    for doc_id in doc_scores:
        if doc_id not in idx_lst:
            score = doc_scores[doc_id]
            result.append((score, doc_id))
    return result


def cal_doc_scores(weight_query: List, weight_doc: Dict[int, List]) -> Dict[int, float]:
    """
    helper function, calculate the scores for each docs
    :param weight_query:
    :param weight_doc:
    :return: a dict with key to be the doc_id and value is the docs_scores calculated with cos similarity
    """
    doc_scores = {}
    dim = len(weight_query)
    for key in weight_doc:
        sim = 0
        for i in range(dim):
            sim += weight_query[i] * weight_doc[key][i]
        doc_scores[key] = sim
    return doc_scores


def query_weight(query_terms: List[str], shelve_index: shelve.Shelf, doc_length_shelve: shelve.Shelf) -> List:
    """
    helper function, find the query weight using logarithmic TF*IDF formula w/o length norm
    :param query_terms: normalized query terms
    :param shelve_index:
    :param doc_length_shelve:
    :return:
    """
    term_set = set(query_terms)
    tf_lst = []
    idf_lst = []
    # count the number of all doc
    N = len(doc_length_shelve)
    for term in term_set:
        # calculate tf
        tf = text_processor.tf(query_terms.count(term))
        tf_lst.append(tf)
        # calculate idf
        idf = text_processor.idf(N, len(shelve_index[term]))
        idf_lst.append(idf)
    # calculate the product for each term
    result = [tf_lst[i] * idf_lst[i] for i in range(len(term_set))]
    return result


def doc_weight(
        query_terms: List[str], shelve_index: shelve.Shelf, doc_length_shelve: shelve.Shelf
) -> Dict[int, List]:
    """
    helper function, find the doc weight using logarithmic TF formula w/ length norm
    :param query_terms:
    :param shelve_index:
    :param doc_length_shelve:
    :return: a dict with the key to be the doc_id, and value is a np.array show the doc_weight
    """
    result = {}
    term_set = list(set(query_terms))
    # dimension of the weight vector
    dim = len(term_set)
    # find all doc with at least one term match
    id_lst = []
    for term in term_set:
        for doc_id, _ in shelve_index[term]:
            id_lst.append(doc_id)
    id_lst = set(id_lst)
    # set all values to
    for doc_id in id_lst:
        result.setdefault(doc_id, [0] * dim)
    # assign value
    for i in range(dim):
        term = term_set[i]
        for doc_id, tf in shelve_index[term]:
            result[doc_id][i] = tf
    # normalize to the length
    for doc_id in result:
        norm = doc_length_shelve[str(doc_id)]
        result[doc_id] = [e / norm for e in result[doc_id]]
    return result


def query_inverted_index(
        query: str, k: int, shelve_index: shelve.Shelf, doc_length_shelve: shelve.Shelf
) -> Tuple[List[Tuple[float, int]], List[str], List[str]]:
    """
    disjunctive query over the shelve_index
    return a list of matched documents (output from the function top_k_docs), a list of stop words and a list of unknown words separately
    :param query:
    :param k:
    :param shelve_index:
    :param doc_length_shelve:
    :return:
    """
    # separate all terms
    query_terms, stopword, unknown = parse_query(query, shelve_index)
    # calculate query and doc weights
    weight_query = query_weight(query_terms, shelve_index, doc_length_shelve)
    weight_doc = doc_weight(query_terms, shelve_index, doc_length_shelve)
    # calculate scores for each doc
    doc_scores = cal_doc_scores(weight_query, weight_doc)
    # choose top_k_docs
    results = top_k_docs(doc_scores, k)
    return results, stopword, unknown


if __name__ == "__main__":
    data_dir = Path("pa4_data")
    wapo_path = data_dir.joinpath("test_corpus.jl")
    build_inverted_index(wapo_path, str(data_dir.joinpath('wapo_shelve')),
                         str(data_dir.joinpath('wapo_vec_norm_shelve')))
