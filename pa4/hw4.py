import shelve
from pathlib import Path
import argparse
from flask import Flask, render_template, request

from text_processing import TextProcessing
from utils import load_wapo
from inverted_index import build_inverted_index, query_inverted_index, parse_query, doc_weight

app = Flask(__name__)

data_dir = Path("pa4_data")
wapo_path = data_dir.joinpath("wapo_pa4.jl")
# wapo_path = data_dir.joinpath("test_corpus.jl")
wapo_docs = {
    doc["doc_id"]: doc for doc in load_wapo(wapo_path)
}  # comment out this line if you use the database
tp = TextProcessing.from_nltk()


# home page
@app.route("/")
def home():
    return render_template("home.html")


# result page
@app.route("/results", methods=["POST"])
def results():
    global result_lst, score_lst, idf_score, term_lst, query_text, num, page_num, stopwords, unknowns, hits
    N = len(db_norm)
    result_lst = []
    # the list contains the cos similarity scores for each doc
    score_lst = []
    # the list contains the idf scores for each term, [(term1, idf1), (term2, idf2)]
    idf_score = []
    # the list contains the terms that appear in each document [[term1, ..., termm], [term1, ..., termn]]
    term_lst = []
    # Get the raw user query from search bar
    query_text = request.form["query"]
    query_terms, _, _ = parse_query(query_text, db)
    term_set = list(set(query_terms))
    weight_doc = doc_weight(query_terms, db, db_norm)
    for term in query_terms:
        idf = tp.idf(N, len(db[term]))
        idf_score.append((term, round(idf, 4)))
    idf_score = list(set(idf_score))
    idx_lst, stopwords, unknowns = query_inverted_index(query_text, 10, db, db_norm)
    hits = len(idx_lst)
    for score, idx in idx_lst:
        d = wapo_docs[idx]
        result_lst.append(d)
        score_lst.append(round(score, 4))
        terms = [term_set[i] for i in range(len(term_set)) if weight_doc[idx][i] != 0]
        term_lst.append(terms)
    num = len(result_lst)
    page_num = (num - 1) // 8 + 1
    return render_template("results.html", query_text=query_text, result_lst=result_lst, score_lst=score_lst,
                           idf_score=idf_score, term_lst=term_lst, stopwords=stopwords,
                           unknowns=unknowns, hits=hits, num=num, page_num=page_num,
                           page_id=1)  # add variables as you wish


# "next page" to show more results
@app.route("/results/<int:page_id>", methods=["POST"])
def next_page(page_id):
    return render_template("results.html", query_text=query_text, result_lst=result_lst, score_lst=score_lst,
                           idf_score=idf_score, term_lst=term_lst, stopwords=stopwords,
                           unknowns=unknowns, hits=hits, num=num, page_num=page_num,
                           page_id=page_id)  # add variables as you wish


# document page
@app.route("/doc_data/<int:doc_id>")
def doc_data(doc_id):
    doc = wapo_docs[doc_id]
    return render_template("doc.html", doc=doc)  # add variables as you wish


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boolean IR system")
    parser.add_argument("--build")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.build:
        build_inverted_index(
            wapo_path,
            str(data_dir.joinpath(args.build)),
            str(data_dir.joinpath(args.build)) + "_doc_len",
        )
    if args.run:
        db = shelve.open(str(data_dir.joinpath('wapo_shelve')))
        db_norm = shelve.open(str(data_dir.joinpath('wapo_shelve_doc_len')))
        # db = shelve.open(str(data_dir.joinpath('test_shelve')))
        # db_norm = shelve.open(str(data_dir.joinpath('test_shelve_doc_len')))
        app.run(debug=True, port=5000)
