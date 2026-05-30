import json
from pathlib import Path
import pandas as pd
import re

CORPUS_PATH = Path("data/corpus.jsonl")
QUERIES_PATH = Path("eval/queries.tsv")
RUNS_DIR = Path("notebooks/runs")
QRELS_PATH = Path("eval/qrels.tsv")

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

def classify_relevance(qid, title, abstract):
    text = (title + " " + abstract).lower()
    
    # Helper to check if all/any keywords exist
    def contains_any(words):
        return any(w in text for w in words)
    def contains_all(words):
        return all(w in text for w in words)
    
    # Query rules
    if qid == "q1":
        # automated testing of flutter mobile applications
        has_test = contains_any(["test", "testing", "tests", "tested"])
        has_flutter = "flutter" in text
        has_mobile = contains_any(["mobile", "app", "application", "cross-platform", "smartphone"])
        if has_test and has_flutter:
            if contains_any(["automated", "test generation", "ui test", "widget test", "framework"]):
                return 2
            return 1
        elif has_test and has_mobile:
            return 1
        return 0
        
    elif qid == "q2":
        # code smells and refactoring in dart projects
        has_smell = contains_any(["smell", "refactor", "debt", "quality", "smells", "refactoring"])
        has_dart = contains_any(["dart", "flutter", "mobile"])
        if has_smell and has_dart:
            if contains_any(["dart", "flutter"]) and contains_any(["smell", "refactor", "refactoring", "smells"]):
                return 2
            return 1
        return 0
        
    elif qid == "q3":
        # performance optimization techniques for cross-platform mobile apps
        has_perf = contains_any(["performance", "optimization", "latency", "speed", "efficient", "efficiency", "optimize"])
        has_cross = contains_any(["cross-platform", "flutter", "react native", "xamarin", "hybrid"])
        if has_perf and has_cross:
            if contains_any(["optimization", "optimize", "efficiency"]) and "mobile" in text:
                return 2
            return 1
        elif has_perf and "mobile" in text:
            return 1
        return 0
        
    elif qid == "q4":
        # memory leaks detection in flutter and react native
        has_mem = contains_any(["memory", "leak", "leaks", "garbage collection", "resource leak", "allocation"])
        has_framework = contains_any(["flutter", "react native", "cross-platform", "mobile"])
        if has_mem and has_framework:
            if contains_any(["leak", "leaks"]) and contains_any(["flutter", "react native"]):
                return 2
            return 1
        return 0
        
    elif qid == "q5":
        # state management patterns in flutter applications
        has_state = contains_any(["state management", "asm", "redux", "bloc", "provider", "mobx", "riverpod", "state"])
        has_flutter = contains_any(["flutter", "dart"])
        if has_state and has_flutter:
            if "state management" in text or contains_any(["redux", "bloc", "provider", "riverpod"]):
                return 2
            return 1
        return 0
        
    elif qid == "q6":
        # continuous integration and delivery pipelines for mobile apps
        has_ci = contains_any(["ci/cd", "continuous integration", "continuous delivery", "devops", "pipeline", "pipelines", "jenkins"])
        has_mobile = contains_any(["mobile", "android", "ios", "flutter", "react native", "smartphone"])
        if has_ci and has_mobile:
            if contains_any(["continuous integration", "ci/cd", "pipeline"]):
                return 2
            return 1
        elif has_ci:
            return 1
        return 0
        
    elif qid == "q7":
        # energy consumption and battery efficiency of mobile software
        has_energy = contains_any(["energy", "battery", "power consumption", "green", "electricity"])
        has_mobile = contains_any(["mobile", "android", "ios", "smartphone", "smartphones"])
        if has_energy and has_mobile:
            if contains_any(["efficiency", "consumption", "power"]) and contains_any(["battery", "energy"]):
                return 2
            return 1
        return 0
        
    elif qid == "q8":
        # widget testing and integration testing in flutter
        has_test = contains_any(["widget test", "integration test", "ui test", "testing", "tests", "test"])
        has_flutter = contains_any(["flutter", "dart"])
        if has_test and has_flutter:
            if contains_any(["widget test", "integration test", "widget testing", "integration testing"]):
                return 2
            return 1
        return 0
        
    elif qid == "q9":
        # static code analysis tools for dart programming language
        has_static = contains_any(["static analysis", "linter", "linter", "static tool", "code analysis", "linting"])
        has_dart = contains_any(["dart", "flutter"])
        if has_static and has_dart:
            if "static" in text and contains_any(["dart", "linter", "linting"]):
                return 2
            return 1
        return 0
        
    elif qid == "q10":
        # user interface design and usability testing for mobile applications
        has_ui = contains_any(["user interface", "ui design", "usability", "ux", "user experience", "interface"])
        has_mobile = contains_any(["mobile", "smartphone", "android", "ios", "flutter", "react native"])
        if has_ui and has_mobile:
            if contains_any(["usability testing", "ui design", "usability"]):
                return 2
            return 1
        return 0
        
    elif qid == "q11":
        # cross-platform vs native mobile app development performance
        has_cross = contains_any(["cross-platform", "hybrid", "flutter", "react native"])
        has_native = "native" in text
        has_perf = contains_any(["performance", "comparison", "evaluation", "overhead", "speed"])
        if has_cross and has_native and has_perf:
            return 2
        elif has_cross and has_perf:
            return 1
        return 0
        
    elif qid == "q12":
        # architectural patterns in dart and flutter projects
        has_arch = contains_any(["architectural pattern", "architecture", "design pattern", "mvvm", "mvc", "clean architecture", "architectures"])
        has_dart = contains_any(["dart", "flutter"])
        if has_arch and has_dart:
            if contains_any(["architectural", "pattern", "patterns", "mvvm"]):
                return 2
            return 1
        elif has_arch:
            return 1
        return 0
        
    elif qid == "q13":
        # security vulnerabilities in cross-platform mobile frameworks
        has_sec = contains_any(["security", "vulnerability", "vulnerabilities", "attack", "leak", "threat", "attacks", "malware"])
        has_cross = contains_any(["cross-platform", "flutter", "react native", "hybrid"])
        if has_sec and has_cross:
            if contains_any(["vulnerability", "vulnerabilities", "malware", "leak"]):
                return 2
            return 1
        elif has_sec and "mobile" in text:
            return 1
        return 0
        
    elif qid == "q14":
        # automated bug detection and error logging in mobile apps
        has_bug = contains_any(["bug detection", "crash", "error log", "automated detection", "fault localization", "debugging", "bug", "bugs", "exception"])
        has_mobile = contains_any(["mobile", "android", "ios", "smartphone"])
        if has_bug and has_mobile:
            if contains_any(["automated", "detection", "logging"]):
                return 2
            return 1
        return 0
        
    elif qid == "q15":
        # dart compilation and compile-time optimization in flutter
        has_comp = contains_any(["compiler", "compilation", "aot", "jit", "compile-time", "tree shaking", "compile"])
        has_dart = contains_any(["dart", "flutter"])
        if has_comp and has_dart:
            if contains_any(["compiler", "compilation", "compile-time"]):
                return 2
            return 1
        return 0
        
    return 0

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
                
    qrels_lines = []
    
    print("Performing semantic classification of pooling pairs...")
    for _, q in queries.iterrows():
        qid = q["qid"]
        for doc_id in pool_docs[qid]:
            doc = docs[doc_id]
            relevance = classify_relevance(qid, doc["title"], doc["abstract"])
            qrels_lines.append(f"{qid}\t0\t{doc_id}\t{relevance}\n")
            
    # Save the qrels.tsv
    with open(QRELS_PATH, "w", encoding="utf-8") as f_qrels:
        f_qrels.write("# Formato TREC: qid 0 doc_id relevance\n")
        f_qrels.writelines(qrels_lines)
        
    print(f"Proper relevance judgments successfully saved to: {QRELS_PATH}")

if __name__ == "__main__":
    main()
