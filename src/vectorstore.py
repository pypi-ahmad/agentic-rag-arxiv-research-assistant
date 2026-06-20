"""
src/vectorstore.py — Swappable vector store layer (ChromaDB and Pinecone)

Why a shared interface?
    NB04 builds Graph RAG on top of a vector store. The retrieval logic
    (embed → search → return chunks) is identical whether you're running
    locally with ChromaDB or in production with Pinecone. A single
    VectorStore ABC keeps NB04 store-agnostic — swap one line to go from
    local dev to cloud deployment.

Two implementations:

  1. ChromaVectorStore  — persistent local store backed by DuckDB+Parquet.
       No API key. Zero config. Data survives process restarts.
       Use this for development, tutorials, and single-machine workloads.

  2. PineconeVectorStore — cloud-native serverless index.
       Requires PINECONE_API_KEY env var. Horizontally scalable.
       Use this for production or when sharing the index across machines.

Both are drop-in replacements for each other — the rest of NB04 is identical.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger


# ── Abstract interface ────────────────────────────────────────────────────────

class VectorStore(ABC):
    """Minimal interface shared by all vector store backends."""

    @abstractmethod
    def upsert(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Insert or update chunks and their embeddings.

        Args:
            chunks:     List of chunk dicts (must include "chunk_id" key).
            embeddings: Float32 ndarray of shape (len(chunks), embed_dim).
                        Must already be L2-normalised if using cosine metric.
        """

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[dict]:
        """
        Return the k most similar chunks to query_embedding.

        Args:
            query_embedding: Shape (embed_dim,) or (1, embed_dim). L2-normalised.
            k:               Number of results.

        Returns:
            List of chunk dicts, each augmented with a "score" key (cosine sim).
        """

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors currently stored."""

    @abstractmethod
    def is_empty(self) -> bool:
        """Return True if the store has no vectors."""


# ── ChromaDB — local persistent store ─────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    """
    Persistent vector store backed by ChromaDB (DuckDB + Parquet on disk).

    Why ChromaDB over FAISS for NB04?
        1. Persistence — survives process restarts without re-indexing.
           FAISS requires explicit save/load; ChromaDB is always-on.
        2. Metadata filtering — query by paper category, date, or any
           field stored alongside the vector (future-proofs multi-collection use).
        3. No dependency on a saved index file format — easier to share.

    The collection is created with "hnsw:space": "cosine" so the similarity
    score returned is cosine similarity (1.0 = identical, 0.0 = orthogonal).
    Our embeddings are L2-normalised before insertion, so inner product ==
    cosine similarity — either metric gives identical ranking.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_dir:     Directory where ChromaDB writes its data files.
    """

    def __init__(
        self,
        collection_name: str = "arxiv_graph_rag",
        persist_dir: str | Path = "artifacts/chromadb",
    ) -> None:
        import chromadb

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{collection_name}' at {persist_dir} "
            f"({self._collection.count()} vectors)"
        )

    def upsert(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Insert or update vectors in batches of 500 (ChromaDB limit per call).

        Metadata stored per chunk: paper_id, title, category, chunk_idx.
        The full text is stored as the "document" field.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
            )

        batch_size = 500
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embs = embeddings[start : start + batch_size]

            self._collection.upsert(
                ids=[c["chunk_id"] for c in batch_chunks],
                embeddings=batch_embs.tolist(),
                documents=[c["text"] for c in batch_chunks],
                metadatas=[
                    {
                        "paper_id":  c.get("paper_id", ""),
                        "title":     c.get("title", "")[:200],   # Chroma metadata limit
                        "category":  c.get("category", ""),
                        "chunk_idx": str(c.get("chunk_idx", 0)),
                    }
                    for c in batch_chunks
                ],
            )
        logger.info(f"ChromaDB upserted {len(chunks)} chunks → total: {self.count()}")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[dict]:
        query_vec = query_embedding.flatten().tolist()
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.count()),
            # `ids` are returned by default in ChromaDB query responses.
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunk = {
                "chunk_id":  cid,
                "paper_id":  meta["paper_id"],
                "title":     meta["title"],
                "category":  meta["category"],
                "chunk_idx": int(meta["chunk_idx"]),
                "text":      doc,
                "score":     float(1.0 - dist),   # ChromaDB returns cosine distance
                "retriever": "chromadb",
            }
            chunks.append(chunk)
        return chunks

    def count(self) -> int:
        return self._collection.count()

    def is_empty(self) -> bool:
        return self.count() == 0


# ── Pinecone — cloud serverless store ────────────────────────────────────────

class PineconeVectorStore(VectorStore):
    """
    Cloud vector store backed by Pinecone serverless.

    Why Pinecone for production?
        1. No infrastructure to manage — serverless scales to billions of vectors.
        2. Shareable — the same index is accessible from any machine or service.
        3. Built-in metadata filtering, namespaces, and hybrid search (sparse+dense).

    Setup:
        1. Sign up at pinecone.io (free tier supports up to 100K vectors)
        2. Create a project and copy your API key
        3. Set the env var: export PINECONE_API_KEY="pc-..."

    The index is created automatically on first use if it doesn't already exist.
    Dimension is inferred from the first batch of embeddings.

    Args:
        index_name: Name of the Pinecone index (created if absent).
        api_key:    Pinecone API key. Defaults to PINECONE_API_KEY env var.
        dimension:  Embedding dimension (2560 for qwen3-embedding:4b).
        cloud:      Cloud provider for serverless index. Default: "aws".
        region:     Region for serverless index. Default: "us-east-1".
    """

    def __init__(
        self,
        index_name: str = "agentic-rag-arxiv",
        api_key: str | None = None,
        dimension: int = 2560,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        from pinecone import Pinecone, ServerlessSpec

        api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY not set. "
                "Export it with: export PINECONE_API_KEY='pc-...'"
            )

        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._dimension = dimension

        # Create index if it doesn't already exist
        existing = [idx.name for idx in self._pc.list_indexes()]
        if index_name not in existing:
            logger.info(f"Creating Pinecone serverless index '{index_name}' (dim={dimension})")
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        else:
            logger.info(f"Using existing Pinecone index '{index_name}'")

        self._index = self._pc.Index(index_name)
        stats = self._index.describe_index_stats()
        logger.info(f"Pinecone index '{index_name}': {stats.total_vector_count} vectors")

    def upsert(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        Upsert vectors in batches of 100 (Pinecone recommended batch size).

        Metadata stored per vector: paper_id, title, category, text (first 400 chars).
        Pinecone metadata values must be strings, numbers, or booleans.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have equal length"
            )

        batch_size = 100
        total_upserted = 0

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embs = embeddings[start : start + batch_size]

            vectors = [
                {
                    "id":     c["chunk_id"],
                    "values": emb.tolist(),
                    "metadata": {
                        "paper_id":  c.get("paper_id", ""),
                        "title":     c.get("title", "")[:200],
                        "category":  c.get("category", ""),
                        "chunk_idx": int(c.get("chunk_idx", 0)),
                        "text":      c.get("text", "")[:400],   # Pinecone metadata size limit
                    },
                }
                for c, emb in zip(batch_chunks, batch_embs)
            ]
            self._index.upsert(vectors=vectors)
            total_upserted += len(vectors)

        logger.info(f"Pinecone upserted {total_upserted} vectors → total: {self.count()}")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[dict]:
        query_vec = query_embedding.flatten().tolist()
        result = self._index.query(
            vector=query_vec,
            top_k=k,
            include_metadata=True,
        )

        chunks = []
        for match in result.matches:
            meta = match.metadata or {}
            chunk = {
                "chunk_id":  match.id,
                "paper_id":  meta.get("paper_id", ""),
                "title":     meta.get("title", ""),
                "category":  meta.get("category", ""),
                "chunk_idx": int(meta.get("chunk_idx", 0)),
                "text":      meta.get("text", ""),
                "score":     float(match.score),
                "retriever": "pinecone",
            }
            chunks.append(chunk)
        return chunks

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.total_vector_count

    def is_empty(self) -> bool:
        return self.count() == 0
