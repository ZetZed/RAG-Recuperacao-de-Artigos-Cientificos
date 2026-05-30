import json
from pathlib import Path
import pandas as pd
import re

CORPUS_PATH = Path("data/corpus.jsonl")
QUERIES_PATH = Path("eval/queries.tsv")
RUNS_DIR = Path("notebooks/runs")
QRELS_PATH = Path("eval/qrels.tsv")
POOL_TXT_PATH = Path("eval/pooling_to_annotate.txt")

def read_run_top_k(path: Path, k: int):
    run = {}
    if not path.exists():
        return run
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, _, _ = parts
            rank = int(rank)
            if rank <= k:
                if qid not in run:
                    run[qid] = []
                run[qid].append(docid)
    return run

def get_word_set(text: str):
    return set(re.findall(r"\b\w{3,}\b", text.lower()))

def main():
    print("Loading corpus...")
    docs = {}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["arxiv_id"]] = d
            
    print("Loading queries...")
    queries = pd.read_csv(QUERIES_PATH, sep="\t", names=["qid", "text"])
    
    # Load top 10 from each run to perform pooling
    runs = ["bm25.trec", "knn.trec", "hybrid.trec"]
    pool_docs = {q["qid"]: set() for _, q in queries.iterrows()}
    
    for run_file in runs:
        run_data = read_run_top_k(RUNS_DIR / run_file, k=10)
        for qid, doc_list in run_data.items():
            if qid in pool_docs:
                pool_docs[qid].update(doc_list)
                
    total_pairs = sum(len(docs) for docs in pool_docs.values())
    print(f"Total pooling pairs to evaluate: {total_pairs}")
    
    qrels_lines = []
    
    # Write helper file for manual annotation and generate draft qrels
    with open(POOL_TXT_PATH, "w", encoding="utf-8") as f_txt:
        f_txt.write("ARQUIVO DE APOIO À ANOTAÇÃO MANUAL DE RELEVÂNCIA (POOLING)\n")
        f_txt.write("========================================================\n\n")
        
        for _, q in queries.iterrows():
            qid = q["qid"]
            query_text = q["text"]
            f_txt.write(f"QUERY: {qid} - \"{query_text}\"\n")
            f_txt.write("-" * 80 + "\n")
            
            query_words = get_word_set(query_text)
            
            # Sort documents in pool by simple term overlap score for presentation
            pool_list = list(pool_docs[qid])
            doc_scores = []
            for doc_id in pool_list:
                doc = docs.get(doc_id)
                if not doc:
                    continue
                doc_text = (doc["title"] + " " + doc["abstract"]).lower()
                doc_words = get_word_set(doc_text)
                overlap = len(query_words.intersection(doc_words))
                doc_scores.append((doc_id, overlap))
                
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            for doc_id, overlap_score in doc_scores:
                doc = docs[doc_id]
                title = doc["title"]
                abstract = doc["abstract"]
                
                # Heuristic draft relevance:
                # If term overlap score is high, set relevance to 1 or 2
                # Query 1: automated, testing, flutter, mobile, applications -> needs 'test' and 'flutter'
                # Let's check overlap_score relative to query size
                relevance = 0
                if overlap_score >= 4:
                    relevance = 2
                elif overlap_score >= 2:
                    relevance = 1
                else:
                    relevance = 0
                
                # Format to qrels line: qid <tab> 0 <tab> doc_id <tab> relevance
                qrels_lines.append(f"{qid}\t0\t{doc_id}\t{relevance}\n")
                
                f_txt.write(f"Document ID: {doc_id}\n")
                f_txt.write(f"Title: {title}\n")
                f_txt.write(f"Abstract: {abstract}\n")
                f_txt.write(f"Draft Relevance Label (overlap={overlap_score}): {relevance} (0=Não-Relevante, 1=Relevante, 2=Altamente Relevante)\n")
                f_txt.write(f"Sua Anotação final: [ ]\n")
                f_txt.write("." * 80 + "\n")
            f_txt.write("\n" + "="*80 + "\n\n")
            
    # Save the draft qrels.tsv
    with open(QRELS_PATH, "w", encoding="utf-8") as f_qrels:
        f_qrels.write("# Formato TREC: qid 0 doc_id relevance\n")
        f_qrels.writelines(qrels_lines)
        
    print(f"Human-readable pooling file saved to: {POOL_TXT_PATH}")
    print(f"Draft relevance judgments saved to: {QRELS_PATH}")

if __name__ == "__main__":
    main()
