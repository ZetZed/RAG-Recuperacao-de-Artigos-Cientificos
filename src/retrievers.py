import json
import os
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from preprocessing import TextPreprocessor

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75, mode="simple"):
        self.k1 = k1
        self.b = b
        self.preprocessor = TextPreprocessor(mode=mode)
        self.bm25 = None
        self.docs = []

    def fit(self, docs: list[dict]):
        self.docs = docs
        # Concatenate title and abstract
        texts = [d["title"] + ". " + d["abstract"] for d in docs]
        tokenized_corpus = [self.preprocessor.preprocess(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, k: int = 100) -> list[tuple[int, float, dict]]:
        q_tokens = self.preprocessor.preprocess(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i]), self.docs[i]) for i in top_idx]


class DenseKNNRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="data"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.docs = []
        self.doc_embeddings = None
        self.embeddings_path = self.cache_dir / f"doc_embeddings_{model_name}.npy"

    def _lazy_load_model(self):
        if self.model is None:
            print(f"Loading SentenceTransformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def fit(self, docs: list[dict], force_recompute=False):
        self.docs = docs
        
        if not force_recompute and self.embeddings_path.exists():
            print(f"Loading cached document embeddings from {self.embeddings_path}...")
            self.doc_embeddings = np.load(self.embeddings_path)
            # Ensure shape matches docs count
            if self.doc_embeddings.shape[0] != len(docs):
                print("Cache shape mismatch. Recomputing embeddings...")
                force_recompute = True
                
        if force_recompute or self.doc_embeddings is None:
            self._lazy_load_model()
            print("Encoding document corpus (this can take a minute)...")
            texts = [d["title"] + ". " + d["abstract"] for d in docs]
            # Encode in batches
            self.doc_embeddings = self.model.encode(
                texts, 
                show_progress_bar=True, 
                batch_size=32,
                convert_to_numpy=True
            )
            # L2 normalize embeddings for easier cosine similarity
            norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
            self.doc_embeddings = self.doc_embeddings / (norms + 1e-9)
            
            # Save to cache
            np.save(self.embeddings_path, self.doc_embeddings)
            print(f"Embeddings saved to cache: {self.embeddings_path}")

    def search(self, query: str, k: int = 100) -> list[tuple[int, float, dict]]:
        self._lazy_load_model()
        # Encode query
        q_emb = self.model.encode(query, convert_to_numpy=True)
        # Normalize
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        
        # Cosine similarity is simple dot product if normalized
        similarities = np.dot(self.doc_embeddings, q_norm)
        
        top_idx = similarities.argsort()[::-1][:k]
        return [(int(i), float(similarities[i]), self.docs[i]) for i in top_idx]


class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, knn_retriever: DenseKNNRetriever):
        self.bm25_retriever = bm25_retriever
        self.knn_retriever = knn_retriever

    def search(self, query: str, k: int = 100, rrf_constant: int = 60) -> list[tuple[str, float, dict]]:
        """
        Combines results of BM25 and KNN using Reciprocal Rank Fusion (RRF).
        RRF Score(d) = 1 / (c + rank_bm25(d)) + 1 / (c + rank_knn(d))
        """
        # Retrieve larger candidate sets (e.g. 2x k) to ensure overlap
        bm25_res = self.bm25_retriever.search(query, k=k * 2)
        knn_res = self.knn_retriever.search(query, k=k * 2)
        
        scores = {}
        docs_map = {}
        
        # Rank from 1 to N
        for rank, (_, _, doc) in enumerate(bm25_res, 1):
            doc_id = doc["arxiv_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank))
            docs_map[doc_id] = doc
            
        for rank, (_, _, doc) in enumerate(knn_res, 1):
            doc_id = doc["arxiv_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_constant + rank))
            docs_map[doc_id] = doc
            
        # Sort by RRF score descending
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [(doc_id, score, docs_map[doc_id]) for doc_id, score in sorted_docs]
