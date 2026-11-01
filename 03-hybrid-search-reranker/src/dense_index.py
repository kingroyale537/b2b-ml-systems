"""Dense Vector Retriever using FAISS HNSW and Transformer Embeddings."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

# Prevent duplicate libomp crash on macOS when combining PyTorch and FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data import SearchDocument


class DenseHNSWIndex:
    """FAISS HNSW Vector Index for dense semantic similarity search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.m = m

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.to(self.device)
        self.encoder.eval()

        # Initialize FAISS HNSW index with Inner Product (MIPS) on normalized vectors
        self.faiss_index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.hnsw.efConstruction = ef_construction
        self.faiss_index.hnsw.efSearch = ef_search

        self.doc_ids: List[str] = []
        self.corpus_docs: List[SearchDocument] = []

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode list of strings into L2-normalized float32 numpy array."""
        embeddings_list = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)

                outputs = self.encoder(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                pooled = self._mean_pooling(outputs, inputs["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                embeddings_list.append(normalized.cpu().numpy().astype(np.float32))

        return np.vstack(embeddings_list)

    def index(self, documents: List[SearchDocument]) -> float:
        """Index all documents into the FAISS HNSW graph."""
        t0 = time.perf_counter()
        self.corpus_docs = documents
        self.doc_ids = [d.doc_id for d in documents]

        texts = [d.to_index_text() for d in documents]
        vectors = self.encode_texts(texts)

        self.faiss_index.add(vectors)
        return time.perf_counter() - t0

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Search top_k nearest documents using MIPS."""
        if self.faiss_index.ntotal == 0:
            return []

        query_vec = self.encode_texts([query])
        k = min(top_k, self.faiss_index.ntotal)

        distances, indices = self.faiss_index.search(query_vec, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.doc_ids):
                results.append((self.doc_ids[idx], float(dist)))

        return results
