from typing import List, Tuple, Union

import faiss
import numpy as np
import torch

from roll.pipeline.agentic.memory.memory_config import SearcherConfig
from roll.pipeline.agentic.memory.searcher.base_searcher import BaseSearcher
from roll.utils.logging import get_logger

logger = get_logger()


class FaissEmbeddingSearcher(BaseSearcher):
    """
    FAISS-based approximate nearest neighbor searcher with GPU support.

    Features:
    - Automatic index selection based on memory size
    - GPU acceleration when available
    - Batch query support for efficient processing
    - Assumes embeddings are already L2 normalized
    """

    def __init__(self, memory_config: SearcherConfig):
        super().__init__(memory_config)
        self.dimension = None
        self.index = None
        self.uid_mapping = []  # Maps index position → uid
        self.rebuild_threshold = getattr(
            memory_config, "faiss_rebuild_threshold", 100
        )
        self.updates_since_rebuild = 0

        # GPU support
        self.use_gpu = torch.cuda.is_available()
        self.gpu_resources = None
        self.gpu_device_id = 0
        if self.use_gpu:
            try:
                self.gpu_device_id = torch.cuda.current_device()
            except Exception:
                # torch may be available but no device selected yet
                torch.cuda.init()
                self.gpu_device_id = torch.cuda.current_device()

            try:
                self.gpu_resources = faiss.StandardGpuResources()
                num_gpus = faiss.get_num_gpus()
                logger.info(
                    "FAISS GPU support enabled "
                    f"(visible_device={self.gpu_device_id}, total_gpus={num_gpus})"
                )
            except Exception as e:
                logger.warning(
                    f"FAISS GPU initialization failed: {e}, falling back to CPU"
                )
                self.use_gpu = False

    def _build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index based on memory size.

        Strategy:
        - N < 100K: Flat index (exact search) with GPU
        - 100K ≤ N < 1M: IVF index (approximate) with GPU
        - N ≥ 1M: HNSW index (approximate) CPU-only

        Note: Assumes embeddings are already L2 normalized
        """
        N, D = embeddings.shape

        if self.dimension is None:
            self.dimension = D
            logger.info(f"FAISS index dimension: {D}")

        # Strategy 1: Flat index for small-to-medium datasets
        if N < 100000:
            self.index = faiss.IndexFlatIP(D)
            if self.use_gpu and self.gpu_resources:
                try:
                    self.index = faiss.index_cpu_to_gpu(
                        self.gpu_resources, self.gpu_device_id, self.index
                    )
                    logger.info(
                        f"Built FAISS Flat index on GPU for {N} entries"
                    )
                except Exception as e:
                    logger.warning(
                        f"GPU index creation failed: {e}, using CPU"
                    )
            else:
                logger.info(
                    f"Built FAISS Flat index on CPU for {N} entries"
                )

        # Strategy 2: IVF index for medium-to-large datasets
        elif N < 1000000:
            nlist = int(np.sqrt(N))  # Number of clusters
            quantizer = faiss.IndexFlatIP(D)
            self.index = faiss.IndexIVFFlat(
                quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = 10  # Search in 10 nearest clusters

            if self.use_gpu and self.gpu_resources:
                try:
                    self.index = faiss.index_cpu_to_gpu(
                        self.gpu_resources, self.gpu_device_id, self.index
                    )
                    logger.info(
                        f"Built FAISS IVF index on GPU for {N} entries "
                        f"(nlist={nlist}, nprobe={self.index.nprobe})"
                    )
                except Exception as e:
                    logger.warning(
                        f"GPU index creation failed: {e}, using CPU"
                    )
            else:
                logger.info(
                    f"Built FAISS IVF index on CPU for {N} entries "
                    f"(nlist={nlist}, nprobe={self.index.nprobe})"
                )

        # Strategy 3: HNSW index for very large datasets (CPU only)
        else:
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(
                D, M, faiss.METRIC_INNER_PRODUCT
            )
            self.index.hnsw.efSearch = 64  # Search quality parameter
            logger.info(
                f"Built FAISS HNSW index on CPU for {N} entries "
                f"(M={M}, efSearch={self.index.hnsw.efSearch})"
            )

        # Add embeddings to index (assumes already normalized)
        self.index.add(embeddings)
        self.updates_since_rebuild = 0

        logger.info(
            f"FAISS index built successfully with {self.index.ntotal} entries"
        )

    def _ensure_index(self, keys: List[Tuple[str, torch.Tensor]]) -> None:
        """
        Ensure the FAISS index stays synchronized with current memory keys.

        Supports incremental append-only updates by adding new embeddings to the
        existing index. If ordering changes (e.g., deletions or reshuffles) or
        the rebuild threshold is exceeded, the entire index is rebuilt.
        """
        if not keys:
            self.index = None
            self.uid_mapping = []
            return

        uids = [uid for uid, _ in keys]

        # Initial build or post-reset
        if self.index is None or not self.uid_mapping:
            embeddings_np = np.stack(
                [emb.cpu().numpy() for _, emb in keys]
            ).astype("float32")
            self._build_index(embeddings_np)
            self.uid_mapping = uids
            return

        # Fast path: identical ordering
        if len(uids) == len(self.uid_mapping) and uids == self.uid_mapping:
            return

        # Append-only growth: add new embeddings without a rebuild
        if (
            len(uids) > len(self.uid_mapping)
            and uids[: len(self.uid_mapping)] == self.uid_mapping
            and self.updates_since_rebuild <= self.rebuild_threshold
        ):
            new_keys = keys[len(self.uid_mapping) :]
            if new_keys:
                embeddings_np = np.stack(
                    [emb.cpu().numpy() for _, emb in new_keys]
                ).astype("float32")
                self.index.add(embeddings_np)
                self.uid_mapping.extend([uid for uid, _ in new_keys])
                self.updates_since_rebuild = 0
            return

        # Otherwise, fall back to rebuilding index to stay consistent
        embeddings_np = np.stack(
            [emb.cpu().numpy() for _, emb in keys]
        ).astype("float32")
        self._build_index(embeddings_np)
        self.uid_mapping = uids

    def _get_search_result(
        self,
        query_embedding: torch.Tensor,
        keys: List[Tuple[str, torch.Tensor]],
    ) -> List[str]:
        """
        Search for single query embedding.

        Args:
            query_embedding: Query embedding tensor (D,) or (1, D)
            keys: List of (uid, embedding) tuples

        Returns:
            List of top-k UIDs
        """
        if not keys:
            return []

        # Ensure index is in sync with memory keys
        self._ensure_index(keys)

        if self.index is None or not self.uid_mapping:
            return []

        # Prepare query (assume already normalized)
        query_np = (
            query_embedding.cpu().numpy().astype("float32").reshape(1, -1)
        )

        # Search: O(log N) for HNSW, O(sqrt(N)) for IVF
        k = min(self.fetch_num, len(keys))
        distances, indices = self.index.search(query_np, k)

        # Map indices back to UIDs
        top_uids = [
            self.uid_mapping[idx]
            for idx in indices[0]
            if idx < len(self.uid_mapping)
        ]

        return top_uids

    def _get_search_result_batch(
        self,
        query_embeddings: Union[torch.Tensor, np.ndarray],
        keys: List[Tuple[str, torch.Tensor]],
    ) -> List[List[str]]:
        """
        Search for batch of query embeddings efficiently.

        This is significantly faster than calling _get_search_result in a loop
        because FAISS can process multiple queries in parallel.

        Args:
            query_embeddings: Batch of query embeddings (B, D) where B is batch size
            keys: List of (uid, embedding) tuples

        Returns:
            List of lists, where each inner list contains top-k UIDs for one query

        Example:
            >>> query_embeddings = torch.randn(16, 768)  # 16 queries
            >>> results = searcher._get_search_result_batch(query_embeddings, keys)
            >>> len(results)  # 16
            >>> len(results[0])  # top_k UIDs for first query
        """
        if not keys:
            return [[] for _ in range(len(query_embeddings))]

        # Ensure index is in sync with memory keys
        self._ensure_index(keys)

        if self.index is None or not self.uid_mapping:
            return [[] for _ in range(len(query_embeddings))]

        # Prepare batch queries (assume already normalized)
        if isinstance(query_embeddings, torch.Tensor):
            queries_np = query_embeddings.cpu().numpy().astype("float32")
        else:
            queries_np = query_embeddings.astype("float32")

        # Ensure 2D shape (B, D)
        if queries_np.ndim == 1:
            queries_np = queries_np.reshape(1, -1)

        # Batch search: FAISS processes all queries in parallel
        k = min(self.fetch_num, len(keys))
        batch_size = queries_np.shape[0]

        distances, indices = self.index.search(queries_np, k)
        # distances: (B, k), indices: (B, k)

        # Map indices back to UIDs for each query
        batch_results = []
        for i in range(batch_size):
            top_uids = [
                self.uid_mapping[idx]
                for idx in indices[i]
                if idx >= 0 and idx < len(self.uid_mapping)
            ]
            batch_results.append(top_uids)

        return batch_results

    def notify_update(self):
        """Call after memory updates to trigger eventual index rebuild"""
        self.updates_since_rebuild += 1
