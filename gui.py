import os
import sys
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, request

# Adiciona a pasta src ao path para importar os modulos modulares
sys.path.append(str(Path(__file__).parent / "src"))

from retrievers import BM25Retriever, DenseKNNRetriever, HybridRetriever

app = Flask(__name__)

# Variaveis globais para armazenar o corpus e os retrievers indexados
docs = []
bm25 = None
knn = None
hybrid = None

def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[erro] Arquivo do corpus nao encontrado em {path}. Por favor, execute a coleta primeiro.")
        sys.exit(1)
    corpus_docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            corpus_docs.append(json.loads(line))
    return corpus_docs


def init_app():
    global docs, bm25, knn, hybrid
    root_dir = Path(__file__).parent
    corpus_path = root_dir / "data" / "corpus.jsonl"
    
    print("Carregando corpus...")
    docs = load_corpus(corpus_path)
    print(f"Corpus carregado com {len(docs)} artigos.")
    
    print("Inicializando e indexando BM25...")
    bm25 = BM25Retriever()
    bm25.fit(docs)
    
    print("Inicializando KNN Denso (all-MiniLM-L6-v2) e carregando cache...")
    knn = DenseKNNRetriever(cache_dir=str(root_dir / "data"))
    knn.fit(docs)
    
    print("Inicializando Hibrido (RRF)...")
    hybrid = HybridRetriever(bm25, knn)
    print("Todos os recuperadores inicializados com sucesso!")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search")
def search():
    query = request.args.get("query", "").strip()
    k = request.args.get("k", 5, type=int)
    
    if not query:
        return jsonify({"error": "Query vazia"}), 400
        
    try:
        bm25_raw = bm25.search(query, k=k)
        knn_raw = knn.search(query, k=k)
        hybrid_raw = hybrid.search(query, k=k)
        
        # Formata os resultados de forma padronizada
        def format_results(results_list):
            formatted = []
            for rank, item in enumerate(results_list, 1):
                # Desempacota o item com base no formato retornado
                if len(item) == 3:
                    _, score, doc = item
                else:
                    score, doc = item[1], item[2]
                
                formatted.append({
                    "rank": rank,
                    "score": float(score),
                    "title": doc.get("title", ""),
                    "authors": doc.get("authors", []),
                    "categories": doc.get("categories", []),
                    "arxiv_id": doc.get("arxiv_id", ""),
                    "abstract": doc.get("abstract", ""),
                    "url": f"https://arxiv.org/abs/{doc.get('arxiv_id', '')}"
                })
            return formatted

        return jsonify({
            "bm25": format_results(bm25_raw),
            "knn": format_results(knn_raw),
            "hybrid": format_results(hybrid_raw)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_app()
    # Roda o servidor localmente
    app.run(host="127.0.0.1", port=5000, debug=True)
