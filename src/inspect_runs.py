import json
from pathlib import Path

CORPUS_PATH = Path("data/corpus.jsonl")
RUNS_DIR = Path("notebooks/runs")

def load_corpus():
    docs = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["arxiv_id"]] = d
    return docs

def load_run(path: Path):
    run = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts
            if qid not in run:
                run[qid] = []
            run[qid].append((docid, int(rank), float(score)))
    return run

def main():
    docs = load_corpus()
    bm25 = load_run(RUNS_DIR / "bm25.trec")
    knn = load_run(RUNS_DIR / "knn.trec")
    hybrid = load_run(RUNS_DIR / "hybrid.trec")
    
    # Let's inspect q5: "state management patterns in flutter applications"
    q5_query = "state management patterns in flutter applications"
    print(f"=== INSPECTION FOR QUERY q5: '{q5_query}' ===")
    print("\n--- Top 3 BM25 results ---")
    for docid, rank, score in bm25.get("q5", [])[:3]:
        doc = docs[docid]
        print(f"Rank {rank} (Score: {score:.4f}) | ID: {docid}")
        print("Title:", doc["title"])
        print("Abstract snippet:", doc["abstract"][:200] + "...")
        print()
        
    print("\n--- Top 3 KNN results ---")
    for docid, rank, score in knn.get("q5", [])[:3]:
        doc = docs[docid]
        print(f"Rank {rank} (Score: {score:.4f}) | ID: {docid}")
        print("Title:", doc["title"])
        print("Abstract snippet:", doc["abstract"][:200] + "...")
        print()

    # Let's inspect q12: "architectural patterns in dart and flutter projects"
    q12_query = "architectural patterns in dart and flutter projects"
    print(f"\n=== INSPECTION FOR QUERY q12: '{q12_query}' ===")
    print("\n--- Top 3 BM25 results ---")
    for docid, rank, score in bm25.get("q12", [])[:3]:
        doc = docs[docid]
        print(f"Rank {rank} (Score: {score:.4f}) | ID: {docid}")
        print("Title:", doc["title"])
        print("Abstract snippet:", doc["abstract"][:200] + "...")
        print()
        
    print("\n--- Top 3 KNN results ---")
    for docid, rank, score in knn.get("q12", [])[:3]:
        doc = docs[docid]
        print(f"Rank {rank} (Score: {score:.4f}) | ID: {docid}")
        print("Title:", doc["title"])
        print("Abstract snippet:", doc["abstract"][:200] + "...")
        print()

if __name__ == "__main__":
    main()
