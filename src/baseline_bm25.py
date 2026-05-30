import json
import os
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi
from preprocessing import TextPreprocessor

CORPUS_PATH = Path("data/corpus.jsonl")
QUERIES_PATH = Path("eval/queries.tsv")
RUNS_DIR = Path("notebooks/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_PATH = RUNS_DIR / "bm25.trec"

def main():
    print("Loading corpus...")
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents.")

    print("Preprocessing corpus...")
    # Using simple tokenization (no stemming) as standard baseline
    preprocessor = TextPreprocessor(mode="simple")
    
    # Concatenate title and abstract
    texts = [d["title"] + ". " + d["abstract"] for d in docs]
    tokenized_corpus = [preprocessor.preprocess(t) for t in texts]
    
    print("Building BM25 index...")
    # Default parameters k1=1.5, b=0.75
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    print(f"Index built with {len(bm25.idf)} vocabulary terms.")

    print("Loading queries...")
    queries = pd.read_csv(QUERIES_PATH, sep="\t", names=["qid", "text"])
    print(f"Loaded {len(queries)} queries.")

    print("Running searches and saving run to TREC format...")
    with open(RUN_PATH, "w", encoding="utf-8") as f:
        for _, q in queries.iterrows():
            qid = q["qid"]
            query_text = q["text"]
            
            # Tokenize query using same preprocessor
            q_tokens = preprocessor.preprocess(query_text)
            
            # Get scores
            scores = bm25.get_scores(q_tokens)
            
            # Sort scores descending
            top_idx = scores.argsort()[::-1][:100]  # top-100 for evaluation
            
            for rank, idx in enumerate(top_idx, 1):
                doc = docs[idx]
                score = scores[idx]
                f.write(f"{qid} Q0 {doc['arxiv_id']} {rank} {score:.6f} bm25\n")
                
    print(f"Run file saved to: {RUN_PATH}")

if __name__ == "__main__":
    main()
