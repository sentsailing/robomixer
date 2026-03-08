"""FAISS index for fast approximate nearest-neighbor mix embedding search."""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

import faiss
import numpy as np

from robomixer.config import settings

logger = logging.getLogger(__name__)


class FaissIndex:
    """Manages a FAISS inner-product index over L2-normalized mix embeddings.

    Embeddings are assumed to be L2-normalized before insertion, so inner
    product similarity is equivalent to cosine similarity.
    """

    def __init__(
        self,
        dim: int | None = None,
        index_path: Path | None = None,
    ) -> None:
        self.dim = dim or settings.embedding_dim
        self._index_path = index_path or (settings.data_dir / "faiss_mix.index")
        self._id_map_path = self._index_path.with_suffix(".ids")

        self._index = faiss.IndexFlatIP(self.dim)
        self._song_ids: list[UUID] = []

        # Load existing index if available
        if self._index_path.exists():
            self._load()

    @property
    def size(self) -> int:
        """Number of embeddings in the index."""
        return self._index.ntotal

    def add_embedding(self, song_id: UUID, vector: np.ndarray | list[float]) -> None:
        """Add a single embedding to the index.

        The vector should be L2-normalized. If not, it will be normalized.
        """
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._index.add(vec)
        self._song_ids.append(song_id)

    def add_batch(
        self,
        song_ids: list[UUID],
        vectors: np.ndarray,
    ) -> None:
        """Add a batch of embeddings to the index.

        Args:
            song_ids: List of song UUIDs, one per row.
            vectors: 2D array of shape (n, dim), float32.
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        # L2-normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        vecs = vecs / norms
        self._index.add(vecs)
        self._song_ids.extend(song_ids)

    def search(
        self, query_vector: np.ndarray | list[float], k: int = 10
    ) -> list[tuple[UUID, float]]:
        """Search for the k nearest neighbors to a query embedding.

        Args:
            query_vector: Query embedding (will be L2-normalized).
            k: Number of results to return.

        Returns:
            List of (song_id, distance) tuples sorted by descending similarity.
        """
        if self._index.ntotal == 0:
            return []

        vec = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)

        results: list[tuple[UUID, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            results.append((self._song_ids[idx], float(dist)))
        return results

    def remove(self, song_id: UUID) -> bool:
        """Remove an embedding by song ID.

        Since IndexFlatIP doesn't support removal, we rebuild the index
        without the removed entry.
        """
        if song_id not in self._song_ids:
            return False

        idx = self._song_ids.index(song_id)
        # Reconstruct all vectors
        all_vecs = (
            faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self.dim)
            .reshape(self._index.ntotal, self.dim)
            .copy()
        )

        # Remove the entry
        keep_mask = np.ones(len(self._song_ids), dtype=bool)
        keep_mask[idx] = False
        kept_vecs = all_vecs[keep_mask]
        kept_ids = [sid for i, sid in enumerate(self._song_ids) if i != idx]

        # Rebuild
        self._index = faiss.IndexFlatIP(self.dim)
        if len(kept_vecs) > 0:
            self._index.add(kept_vecs)
        self._song_ids = kept_ids
        return True

    def save(self) -> None:
        """Persist the index and ID map to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        self._save_id_map()
        logger.info("Saved FAISS index with %d vectors to %s", self.size, self._index_path)

    def _load(self) -> None:
        """Load the index and ID map from disk."""
        self._index = faiss.read_index(str(self._index_path))
        self._load_id_map()
        logger.info("Loaded FAISS index with %d vectors from %s", self.size, self._index_path)

    def _save_id_map(self) -> None:
        """Save the song_id -> index position mapping."""
        with open(self._id_map_path, "wb") as f:
            for sid in self._song_ids:
                f.write(sid.bytes)

    def _load_id_map(self) -> None:
        """Load the song_id -> index position mapping."""
        self._song_ids = []
        if not self._id_map_path.exists():
            return
        data = self._id_map_path.read_bytes()
        for i in range(0, len(data), 16):
            self._song_ids.append(UUID(bytes=data[i : i + 16]))

    def rebuild_from_db(self, db: Database) -> None:  # noqa: F821
        """Rebuild the full index from all analyses in the database.

        Useful for re-indexing after model retraining.
        """
        from robomixer.storage.db import Database

        if not isinstance(db, Database):
            raise TypeError("Expected a Database instance")

        self._index = faiss.IndexFlatIP(self.dim)
        self._song_ids = []

        songs = db.list_songs()
        for song in songs:
            analysis = db.get_analysis(song.song_id)
            if analysis and analysis.mix_embedding:
                self.add_embedding(song.song_id, analysis.mix_embedding)

        logger.info("Rebuilt FAISS index with %d vectors from database", self.size)
