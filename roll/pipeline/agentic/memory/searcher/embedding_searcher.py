from typing import List, Tuple

import torch

from roll.pipeline.agentic.memory.memory_config import SearcherConfig
from roll.pipeline.agentic.memory.searcher.base_searcher import BaseSearcher
from roll.utils.logging import get_logger

logger = get_logger()


class EmbeddingSearcher(BaseSearcher):
    """
    GPU-based embedding searcher using precomputed embeddings for semantic similarity.

    Design Philosophy:
    -----------------
    - Embeddings are precomputed and stored in memory entries as torch.Tensors (CPU)
    - search_field in keys is expected to be a torch.Tensor (the embedding)
    - Query embeddings are computed in SearchWorker before calling this searcher
    - Searcher only performs cosine similarity computation (no model inference)

    This separation allows:
    - Memory entries to control when/how embeddings are computed
    - Embedding computation to happen in SearchWorker using EmbeddingStrategy
    - Searcher to be a pure computation module (efficient, no model state)
    - Easy testing and debugging
    """

    def __init__(self, memory_config: SearcherConfig):
        super().__init__(memory_config)
        # Determine device for this searcher worker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_search_result(self, query_embedding: torch.Tensor, keys: List[Tuple]) -> List[str]:
        """
        Search using embedding similarity with precomputed embeddings.

        Device Handling:
        ---------------
        1. Query embedding comes from embedding worker (potentially on different GPU)
        2. Memory embeddings stored on CPU (from different embedding workers)
        3. Move both to this searcher's device for computation
        4. This ensures no cross-device tensor operations

        Args:
            query_embedding: Query embedding tensor (from embedding worker, any device)
            keys: List of (uid, embedding_tensor) tuples from memory (stored on CPU)

        Returns:
            List[str]: UIDs of top-k most similar entries
        """
        if not keys:
            return []

        if not isinstance(query_embedding, torch.Tensor):
            logger.error(
                f"query_embedding must be torch.Tensor, got {type(query_embedding)}. "
                "Embedding should be computed in SearchWorker before calling searcher."
            )
            return []

        query_embedding = query_embedding.to(self.device)

        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # 1, D

        query_dtype = query_embedding.dtype
        batched_embeddings = torch.stack([embedding for uid, embedding in keys]).to(
            device=self.device, dtype=query_dtype
        )  # N, D

        similarities = query_embedding @ batched_embeddings.T  # 1, N
        top_k_values, top_k_indices = torch.topk(
            similarities,
            k=min(self.fetch_num, len(keys)),
            dim=1,
            largest=True,
        )

        top_uids = [keys[idx][0] for idx in top_k_indices[0].cpu().tolist()]

        return top_uids

    def _get_search_result_batch(self, query_embeddings: torch.Tensor, keys: List[Tuple]) -> List[List[str]]:
        """
        Batch search using embedding similarity with precomputed embeddings.

        This is significantly faster than calling _get_search_result in a loop
        because it processes all queries in a single GPU operation.

        Args:
            query_embeddings: Batch of query embeddings (B, D) where B is batch size
            keys: List of (uid, embedding_tensor) tuples from memory (stored on CPU)

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

        if not isinstance(query_embeddings, torch.Tensor):
            logger.error(
                f"query_embeddings must be torch.Tensor, got {type(query_embeddings)}. "
                "Embeddings should be computed in SearchWorker before calling searcher."
            )
            return [[] for _ in range(len(query_embeddings))]

        query_embeddings = query_embeddings.to(self.device)

        # Ensure 2D shape (B, D)
        if query_embeddings.dim() == 1:
            query_embeddings = query_embeddings.unsqueeze(0)

        batch_size = query_embeddings.shape[0]
        query_dtype = query_embeddings.dtype

        # Stack all memory embeddings (N, D)
        batched_embeddings = torch.stack([embedding for uid, embedding in keys]).to(
            device=self.device, dtype=query_dtype
        )

        # Batch matrix multiplication: (B, D) @ (D, N) = (B, N)
        similarities = query_embeddings @ batched_embeddings.T  # (B, N)

        # Get top-k for each query in the batch
        k = min(self.fetch_num, len(keys))
        top_k_values, top_k_indices = torch.topk(
            similarities,
            k=k,
            dim=1,
            largest=True,
        )  # (B, k)

        # Map indices back to UIDs for each query
        batch_results = []
        for i in range(batch_size):
            top_uids = [keys[idx][0] for idx in top_k_indices[i].cpu().tolist()]
            batch_results.append(top_uids)

        return batch_results
