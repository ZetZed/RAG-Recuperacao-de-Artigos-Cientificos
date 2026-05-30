import json
from pathlib import Path
import pandas as pd
import sys

# Ensure src/ is in the import path
sys.path.append(str(Path(__file__).parent))
from retrievers import BM25Retriever, DenseKNNRetriever, HybridRetriever

CORPUS_PATH = Path("data/corpus.jsonl")
QUERIES_PATH = Path("eval/queries.tsv")
RUNS_DIR = Path("notebooks/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_PATH = RUNS_DIR / "hybrid.trec"

def main():
    print("Loading corpus...")
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents.")

    # Initialize retrievers
    print("Initializing BM25 retriever...")
    bm25_ret = BM25Retriever(k1=1.5, b=0.75, mode="simple")
    bm25_ret.fit(docs)
    
    print("Initializing KNN retriever...")
    knn_ret = DenseKNNRetriever(model_name="all-MiniLM-L6-v2", cache_dir="data")
    knn_ret.fit(docs) # This should load cached embeddings!
    
    # Initialize hybrid retriever
    print("Initializing Hybrid retriever (RRF)...")
    hybrid_ret = HybridRetriever(bm25_ret, knn_ret)

    print("Loading queries...")
    queries = pd.read_csv(QUERIES_PATH, sep="\t", names=["qid", "text"])
    print(f"Loaded {len(queries)} queries.")

    print("Running hybrid searches and saving run to TREC format...")
    with open(RUN_PATH, "w", encoding="utf-8") as f:
        for _, q in queries.iterrows():
            qid = q["qid"]
            query_text = q["text"]
            
            # Search
            results = hybrid_ret.search(query_text, k=100, rrf_constant=60)
            
            for rank, (doc_id, score, doc) in enumerate(results, 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} hybrid\n")
                
    print(f"Run file saved to: {RUN_PATH}")

if __name__ == "__main__":
    main()
