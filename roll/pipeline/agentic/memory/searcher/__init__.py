from roll.pipeline.agentic.memory.memory_config import SampleStrategy
from roll.pipeline.agentic.memory.searcher.base_searcher import (
    MostRecentSearcher,
    RandomSearcher,
    SimpleSimilaritySearcher,
)
from roll.pipeline.agentic.memory.searcher.embedding_searcher import EmbeddingSearcher


def _create_faiss_embedding_searcher(*args, **kwargs):
    from roll.pipeline.agentic.memory.searcher.faiss_embedding_searcher import FaissEmbeddingSearcher

    return FaissEmbeddingSearcher(*args, **kwargs)


memory_searcher_dict = {
    SampleStrategy.simple_similarity: SimpleSimilaritySearcher,
    SampleStrategy.embedding_similarity: EmbeddingSearcher,
    SampleStrategy.random: RandomSearcher,
    SampleStrategy.most_recent: MostRecentSearcher,
    SampleStrategy.faiss_embedding_similarity: _create_faiss_embedding_searcher,
}
