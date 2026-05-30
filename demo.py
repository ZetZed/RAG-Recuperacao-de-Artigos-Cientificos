#!/usr/bin/env default_api python
"""
demo.py
-------
Script de demonstracao minima de uso para o trabalho pratico de IA (FACOM/UFMS, 2026/1).
Permite executar uma consulta textual e exibe os rankings dos recuperadores
BM25 (esparso), KNN (denso) e Hibrido (RRF) lado a lado.

Uso:
    python demo.py --query "automated testing in flutter"
    ou apenas:
    python demo.py (para modo interativo no terminal)
"""
import sys
import json
import argparse
from pathlib import Path

# Adiciona a pasta src ao path para importar os modulos modulares
sys.path.append(str(Path(__file__).parent / "src"))

from retrievers import BM25Retriever, DenseKNNRetriever, HybridRetriever

def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[erro] Arquivo do corpus nao encontrado em {path}. Por favor, execute a coleta primeiro.")
        sys.exit(1)
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs

def display_results(title: str, results: list, limit: int = 5):
    print(f"\n========================================================")
    print(f"*** TOP {limit} - {title} ***")
    print(f"========================================================")
    for rank, res in enumerate(results[:limit], 1):
        if len(res) == 3:
            # BM25 e KNN retornam (idx, score, doc) ou (doc_id, score, doc)
            _, score, doc = res
        else:
            score, doc = res[1], res[2]
            
        print(f"{rank}. [Score: {score:.4f}] - {doc['title']}")
        print(f"   Autores: {', '.join(doc['authors'][:3])}{' et al.' if len(doc['authors']) > 3 else ''}")
        print(f"   Categorias: {', '.join(doc['categories'])} | Link: https://arxiv.org/abs/{doc['arxiv_id']}")
        print(f"   Abstract (trecho): {doc['abstract'][:180]}...\n")

def main():
    parser = argparse.ArgumentParser(description="Demonstracao interativa dos retrievers.")
    parser.add_argument("--query", type=str, default="", help="Query de busca.")
    parser.add_argument("--k", type=int, default=5, help="Quantidade de resultados a exibir (default: 5).")
    args = parser.parse_args()

    root_dir = Path(__file__).parent
    corpus_path = root_dir / "data" / "corpus.jsonl"
    
    print("Carregando corpus...")
    docs = load_corpus(corpus_path)
    print(f"Corpus carregado com {len(docs)} artigos.")
    
    print("\nInicializando e indexando BM25...")
    bm25 = BM25Retriever()
    bm25.fit(docs)
    
    print("\nInicializando KNN Denso (all-MiniLM-L6-v2) e carregando cache...")
    knn = DenseKNNRetriever(cache_dir=str(root_dir / "data"))
    knn.fit(docs)
    
    print("\nInicializando Hibrido (RRF)...")
    hybrid = HybridRetriever(bm25, knn)
    
    query = args.query.strip()
    if query:
        # Executa query passada por parametro
        print(f"\nExecutando busca para: '{query}'")
        bm25_res = bm25.search(query, k=args.k)
        knn_res = knn.search(query, k=args.k)
        hybrid_res = hybrid.search(query, k=args.k)
        
        display_results("BM25 (Léxico Esparso)", bm25_res, args.k)
        display_results("KNN (Semântico Denso)", knn_res, args.k)
        display_results("Híbrido (RRF)", hybrid_res, args.k)
    else:
        # Loop interativo no terminal
        print("\n=== Sistema Prático de Busca de Artigos Científicos ===")
        print("Digite 'sair' para encerrar o programa.\n")
        while True:
            try:
                query = input("Digite sua busca: ").strip()
                if not query:
                    continue
                if query.lower() in ["sair", "exit", "quit"]:
                    print("Saindo...")
                    break
                
                print(f"\nBuscando: '{query}'...")
                bm25_res = bm25.search(query, k=args.k)
                knn_res = knn.search(query, k=args.k)
                hybrid_res = hybrid.search(query, k=args.k)
                
                display_results("BM25 (Léxico Esparso)", bm25_res, args.k)
                display_results("KNN (Semântico Denso)", knn_res, args.k)
                display_results("Híbrido (RRF)", hybrid_res, args.k)
                print("-" * 60)
            except KeyboardInterrupt:
                print("\nSaindo...")
                break

if __name__ == "__main__":
    main()
