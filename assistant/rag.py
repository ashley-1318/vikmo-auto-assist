"""
assistant/rag.py - Retrieval-Augmented Generation Engine

Design Philosophy:
-----------------
True RAG means we NEVER stuff the entire catalogue into the prompt.
Instead, we:
1. Convert each catalogue row into a semantically rich document.
2. Embed documents using BAAI/bge-small-en-v1.5 (fast, multilingual-capable, well-suited for technical text).
3. Build a FAISS flat-L2 index for exact nearest-neighbor search (appropriate for < 100k docs).
4. At query time, embed the user's query and retrieve only the top-k most relevant products.
5. Those top-k products form the grounded context window for the LLM.

Why BAAI/bge-small-en-v1.5?
- Small model (33M params) → fast inference, low memory
- Outperforms many larger models on retrieval benchmarks (BEIR)
- Trained specifically for asymmetric retrieval (short query vs long passage)
- Stable, well-maintained by BAAI on HuggingFace

Why FAISS (flat L2)?
- Exact search (no approximation) for small catalogues
- In-memory, no server needed
- Persistent via index serialization
- Can upgrade to HNSW/IVF for larger catalogues
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Configure module logger
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "catalogue.csv"
CACHE_DIR = Path(__file__).parent.parent / ".cache"
INDEX_PATH = CACHE_DIR / "faiss.index"
DOCS_PATH = CACHE_DIR / "documents.pkl"
DEFAULT_TOP_K = 5

# BGE models work best with this query prefix (per model card recommendation)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class CatalogueDocument:
    """
    Represents a single catalogue item as a searchable document.

    Design: We flatten all structured fields into a single rich text string
    so the embedding model can capture semantic relationships across fields.
    We also preserve the structured data for tool use and display.
    """

    def __init__(self, row: pd.Series):
        self.sku: str = str(row["sku"])
        self.name: str = str(row["name"])
        self.category: str = str(row["category"])
        self.brand: str = str(row["brand"])
        self.vehicle_fitment: str = str(row["vehicle_fitment"])
        self.price_inr: float = float(row["price_inr"])
        self.stock: int = int(row["stock"])
        self.description: str = str(row["description"])

    def to_text(self) -> str:
        """
        Converts catalogue row to a single rich text document for embedding.

        Strategy: Include all searchable fields in a natural-language format.
        This ensures the embedding captures: part name, category, brand,
        vehicle compatibility, and functional description.
        """
        return (
            f"SKU: {self.sku}. "
            f"Product: {self.name}. "
            f"Category: {self.category}. "
            f"Brand: {self.brand}. "
            f"Compatible Vehicles: {self.vehicle_fitment}. "
            f"Price: INR {self.price_inr}. "
            f"Stock: {self.stock} units. "
            f"Description: {self.description}"
        )

    def to_dict(self) -> Dict:
        """Returns structured dictionary for tool calls and UI display."""
        return {
            "sku": self.sku,
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "vehicle_fitment": self.vehicle_fitment,
            "price_inr": self.price_inr,
            "stock": self.stock,
            "description": self.description,
            "available": self.stock > 0,
            "low_stock": 0 < self.stock <= 10,
        }

    def __repr__(self) -> str:
        return f"<CatalogueDocument sku={self.sku} name={self.name}>"


class RAGEngine:
    """
    Core RAG engine: loads catalogue, builds FAISS index, performs semantic search.

    Lifecycle:
    1. Initialize with catalogue path
    2. load() — reads CSV, builds documents, embeds, creates FAISS index
    3. search(query, top_k) — returns top-k relevant CatalogueDocuments
    """

    def __init__(self, data_path: Path = DEFAULT_DATA_PATH):
        self.data_path = Path(data_path)
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.documents: List[CatalogueDocument] = []
        self.df: Optional[pd.DataFrame] = None
        self._loaded = False

        # Ensure cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def load(self, force_rebuild: bool = False) -> None:
        """
        Load catalogue and build (or restore from cache) the FAISS index.

        Cache strategy: If both the FAISS index and pickled documents exist,
        we skip re-embedding (which takes 30-60s on first run). This speeds up
        subsequent startups dramatically.

        Args:
            force_rebuild: If True, ignore cache and rebuild from scratch.
        """
        logger.info("Loading RAG engine...")

        # Step 1: Load the embedding model (always needed for query-time embedding)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Step 2: Load catalogue CSV
        if not self.data_path.exists():
            raise FileNotFoundError(f"Catalogue not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} catalogue items from {self.data_path}")

        # Step 3: Convert rows to CatalogueDocument objects
        self.documents = [CatalogueDocument(row) for _, row in self.df.iterrows()]

        # Step 4: Load from cache if available, otherwise build index
        if not force_rebuild and INDEX_PATH.exists() and DOCS_PATH.exists():
            self._load_from_cache()
        else:
            self._build_index()

        self._loaded = True
        logger.info("RAG engine ready.")

    def _build_index(self) -> None:
        """
        Embeds all documents and builds the FAISS flat-L2 index.

        Why Flat L2 (exact search)?
        - Our catalogue has < 1000 items — exact search is fast enough.
        - No approximation error means perfect recall.
        - For 10k+ items, switch to IndexIVFFlat or IndexHNSWFlat.
        """
        logger.info("Building FAISS index from scratch (this may take a minute)...")

        # Generate text representations for each document
        texts = [doc.to_text() for doc in self.documents]

        # Embed all documents in batch (faster than one-by-one)
        # Note: BGE models expect passages (not queries) without the prefix
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 norm → cosine similarity via inner product
        )

        embeddings = embeddings.astype(np.float32)
        dimension = embeddings.shape[1]

        # Build FAISS index
        # Using IndexFlatIP (inner product) because we normalized embeddings → cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dimension}")

        # Persist index and documents to cache
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info(f"Index cached to {INDEX_PATH}")

    def _load_from_cache(self) -> None:
        """Restores FAISS index and documents from disk cache."""
        logger.info("Loading FAISS index from cache...")
        self.index = faiss.read_index(str(INDEX_PATH))

        with open(DOCS_PATH, "rb") as f:
            cached_docs = pickle.load(f)

        # Validate cache matches current catalogue
        if len(cached_docs) != len(self.documents):
            logger.warning(
                f"Cache mismatch: {len(cached_docs)} cached vs {len(self.documents)} current. Rebuilding."
            )
            self._build_index()
            return

        self.documents = cached_docs
        logger.info(f"Restored {self.index.ntotal} vectors from cache.")

    def search(
        self, query: str, top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[CatalogueDocument, float]]:
        """
        Semantic search: embeds the query and returns top-k similar documents.

        Args:
            query: Natural language query from the user.
            top_k: Number of results to return.

        Returns:
            List of (CatalogueDocument, similarity_score) tuples, sorted by score.

        Design: We use the BGE query prefix for asymmetric retrieval.
        The prefix conditions the encoder to produce query-appropriate embeddings
        rather than passage embeddings, improving retrieval accuracy.
        """
        if not self._loaded:
            raise RuntimeError("RAGEngine not loaded. Call .load() first.")

        # Apply BGE-recommended query prefix
        prefixed_query = BGE_QUERY_PREFIX + query

        # Embed the query
        query_embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append((self.documents[idx], float(score)))

        return results

    def search_dicts(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Convenience wrapper that returns dicts instead of CatalogueDocument objects.
        Used when formatting results for the LLM context window.
        """
        results = self.search(query, top_k)
        return [
            {**doc.to_dict(), "relevance_score": round(score, 4)}
            for doc, score in results
        ]

    def get_by_sku(self, sku: str) -> Optional[CatalogueDocument]:
        """Direct lookup by SKU (O(n) scan — fine for small catalogues)."""
        for doc in self.documents:
            if doc.sku.upper() == sku.upper():
                return doc
        return None

    def get_all_documents(self) -> List[CatalogueDocument]:
        """Returns all documents (used by tools for exact matching)."""
        return self.documents

    def format_context(self, results: List[Tuple[CatalogueDocument, float]]) -> str:
        """
        Formats retrieved documents into an LLM-ready context string.

        This is what gets injected into the system prompt as grounded context.
        Only this information should be used by the LLM for answers.
        """
        if not results:
            return "No relevant products found in the catalogue."

        lines = ["### Retrieved Product Context (use ONLY this information):"]
        for i, (doc, score) in enumerate(results, 1):
            lines.append(
                f"\n[Product {i}] SKU: {doc.sku}\n"
                f"  Name: {doc.name}\n"
                f"  Category: {doc.category} | Brand: {doc.brand}\n"
                f"  Compatible: {doc.vehicle_fitment}\n"
                f"  Price: ₹{doc.price_inr:,.0f} | Stock: {doc.stock} units\n"
                f"  Description: {doc.description}"
            )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Singleton accessor — ensures the engine is loaded once
# ─────────────────────────────────────────────────────────
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """
    Returns the singleton RAGEngine instance, loading it if necessary.

    Singleton pattern is critical here: loading the embedding model and
    building the FAISS index is expensive. We do it once at startup and
    reuse across all requests.
    """
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        _rag_engine.load()
    return _rag_engine
