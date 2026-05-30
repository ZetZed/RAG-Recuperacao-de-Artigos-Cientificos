import json
from pathlib import Path
import pandas as pd
import sys

# Ensure src/ is in the import path
sys.path.append(str(Path(__file__).parent))
from retrievers import DenseKNNRetriever

CORPUS_PATH = Path("data/corpus.jsonl")
QUERIES_PATH = Path("eval/queries.tsv")
RUNS_DIR = Path("notebooks/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_PATH = RUNS_DIR / "knn.trec"

def main():
    print("Loading corpus...")
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"Loaded {len(docs)} documents.")

    # Initialize retriever
    retriever = DenseKNNRetriever(model_name="all-MiniLM-L6-v2", cache_dir="data")
    
    # Fit (calculates embeddings and caches them)
    retriever.fit(docs)

    print("Loading queries...")
    queries = pd.read_csv(QUERIES_PATH, sep="\t", names=["qid", "text"])
    print(f"Loaded {len(queries)} queries.")

    print("Running searches and saving run to TREC format...")
    with open(RUN_PATH, "w", encoding="utf-8") as f:
        for _, q in queries.iterrows():
            qid = q["qid"]
            query_text = q["text"]
            
            # Search
            results = retriever.search(query_text, k=100)
            
            for rank, (idx, score, doc) in enumerate(results, 1):
                f.write(f"{qid} Q0 {doc['arxiv_id']} {rank} {score:.6f} knn\n")
                
    print(f"Run file saved to: {RUN_PATH}")

if __name__ == "__main__":
    main()
