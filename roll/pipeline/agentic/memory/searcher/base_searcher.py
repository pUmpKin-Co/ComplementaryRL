import json
import math
import os
import random
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from roll.pipeline.agentic.memory.memory_config import SearcherConfig
from roll.pipeline.agentic.memory.memory_structure.base_memory import (
    BaseMemoryStructure,
)
from roll.utils.logging import get_logger

logger = get_logger()


class BaseSearcher(ABC):
    def __init__(self, memory_config: SearcherConfig):
        self.memory_config = memory_config
        self.fetch_num = memory_config.memory_fetch_num

    def search(
        self, query: str, memory: BaseMemoryStructure
    ) -> Tuple[str, List[str]]:
        keys = memory.get_search_info()
        uids = self._get_search_result(query, keys)
        message, _ = memory.search_entry(uids)
        return message, uids

    @abstractmethod
    def _get_search_result(
        self, query: str, keys: List[Tuple[str, Any]]
    ) -> List[str]:
        pass

    def load_state(self) -> None:
        pass

    def save_state(self, path: Optional[str] = None) -> None:
        pass

    def get_state_dict(self) -> dict:
        return {}

    def merge_state_dict(self, state_dict: dict) -> None:
        pass


class RandomSearcher(BaseSearcher):
    def _get_search_result(
        self, query: str, keys: List[Tuple]
    ) -> List[str]:
        keys_num = len(keys)
        if keys_num <= self.fetch_num:
            return [uid for uid, _ in keys]
        else:
            sample_keys = random.sample(keys, self.fetch_num)
            return [uid for uid, _ in sample_keys]


class SimpleSimilaritySearcher(BaseSearcher):
    """
    text-level similarity searcher using TF-IDF with cosine similarity.
    Caches preprocessed text and TF-IDF statistics.
    """

    def __init__(self, memory_config: SearcherConfig):
        super().__init__(memory_config)
        self.memory_searcher_state_path = (
            memory_config.memory_searcher_state_path
        )

        self._text_cache: Dict[str, str] = {}  # uid -> preprocessed_text
        self._tf_cache: Dict[str, Counter] = {}  # uid -> term frequencies
        self._idf_cache: Dict[str, float] = (
            {}
        )  # term -> inverse document frequency
        self._cached_uids: Set[str] = set()  # track which UIDs are cached

        self._df_cache: Dict[str, int] = defaultdict(int)
        self._total_docs = 0

        if self.memory_searcher_state_path and os.path.exists(
            self.memory_searcher_state_path
        ):
            self.load_state(self.memory_searcher_state_path)

    def load_state(self, path: Optional[str] = None) -> None:
        """Load cached statistics from file."""
        if path is None:
            assert (
                self.memory_searcher_state_path is not None
            ), "No searcher state path configured, and no path provided"
            path = self.memory_searcher_state_path

        try:
            with open(path, "r", encoding="utf-8") as f:
                state_data = json.load(f)

            # Restore cached data
            self._text_cache = state_data.get("text_cache", {})
            self._cached_uids = set(state_data.get("cached_uids", []))
            self._total_docs = state_data.get("total_docs", 0)

            # Restore TF cache (convert lists back to Counters)
            tf_cache_data = state_data.get("tf_cache", {})
            self._tf_cache = {
                uid: Counter(tf_dict)
                for uid, tf_dict in tf_cache_data.items()
            }

            # Restore DF cache (convert to defaultdict)
            df_cache_data = state_data.get("df_cache", {})
            self._df_cache = defaultdict(int)
            self._df_cache.update(df_cache_data)

            # Restore IDF cache
            self._idf_cache = state_data.get("idf_cache", {})

            logger.info(
                f"Loaded searcher state with {len(self._cached_uids)} cached entries from {path}"
            )

        except Exception as e:
            logger.error(f"Failed to load searcher state from {path}: {e}")
            # Reset to clean state on load failure
            self.reset()

    def save_state(self, path: Optional[str] = None) -> None:
        """Save cached statistics to file."""
        if path is None:
            assert (
                self.memory_searcher_state_path is not None
            ), "No searcher state path configured, and no path provided"
            path = self.memory_searcher_state_path

        try:
            # Prepare data for serialization
            state_data = {
                "text_cache": self._text_cache,
                "cached_uids": list(self._cached_uids),
                "total_docs": self._total_docs,
                "tf_cache": {
                    uid: dict(counter)
                    for uid, counter in self._tf_cache.items()
                },
                "df_cache": dict(self._df_cache),
                "idf_cache": self._idf_cache,
                "metadata": {"searcher_type": "SimpleSimilaritySearcher"},
            }

            # Ensure directory exists
            state_dir = os.path.dirname(path)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved searcher state with {len(self._cached_uids)} cached entries to {path}"
            )

        except Exception as e:
            logger.error(f"Failed to save searcher state to {path}: {e}")

    def reset(self) -> None:
        """Reset all cached statistics and clear memory."""
        self._text_cache.clear()
        self._tf_cache.clear()
        self._idf_cache.clear()
        self._cached_uids.clear()
        self._df_cache.clear()
        self._total_docs = 0

        logger.info("Reset all cached searcher statistics")

    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase and remove special characters
        text = re.sub(r"[^\w\s]", " ", text.lower())
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text.strip())

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.split()

    def _compute_tf(self, tokens: List[str]) -> Counter:
        """Compute term frequencies."""
        return Counter(tokens)

    def _update_cache(self, keys: List[Tuple]) -> bool:
        """
        Update cache with new entries and remove stale ones.
        Returns True if cache was modified, False otherwise.
        """
        current_uids = {uid for uid, _ in keys}

        # Check if any changes are needed
        stale_uids = self._cached_uids - current_uids
        new_uids = current_uids - self._cached_uids

        if not stale_uids and not new_uids:
            return False  # No changes needed

        cache_modified = False

        # Remove stale entries from cache
        for uid in stale_uids:
            if uid in self._text_cache:
                if uid in self._tf_cache:
                    for term in self._tf_cache[uid]:
                        self._df_cache[term] -= 1
                        if self._df_cache[term] <= 0:
                            del self._df_cache[term]

                del self._text_cache[uid]
                del self._tf_cache[uid]
                self._total_docs -= 1
                cache_modified = True

        # Add new entries to cache
        for uid, search_field in keys:
            if uid in new_uids:
                preprocessed = self._preprocess_text(search_field)
                self._text_cache[uid] = preprocessed

                tokens = self._tokenize(preprocessed)
                tf = self._compute_tf(tokens)
                self._tf_cache[uid] = tf

                for term in set(tokens):
                    self._df_cache[term] += 1

                self._total_docs += 1
                cache_modified = True

        # Update cached UIDs
        self._cached_uids = current_uids

        # Recompute IDF for all terms if cache was modified
        if cache_modified:
            self._update_idf()

        return cache_modified

    def _update_idf(self) -> None:
        """Update inverse document frequency cache."""
        if self._total_docs == 0:
            self._idf_cache.clear()
            return

        for term, df in self._df_cache.items():
            # IDF = log(total_docs / document_frequency)
            self._idf_cache[term] = math.log(self._total_docs / df)

    def _compute_tfidf_vector(self, tf: Counter) -> Dict[str, float]:
        """Compute TF-IDF vector for a document."""
        tfidf = {}
        max_tf = max(tf.values()) if tf else 1

        for term, freq in tf.items():
            # Normalized TF * IDF
            normalized_tf = freq / max_tf
            idf = self._idf_cache.get(term, 0)
            tfidf[term] = normalized_tf * idf

        return tfidf

    def _cosine_similarity(
        self, vec1: Dict[str, float], vec2: Dict[str, float]
    ) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())

        if not common_terms:
            return 0.0

        # Compute dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

        # Compute magnitudes
        mag1 = sum(val**2 for val in vec1.values()) ** 0.5
        mag2 = sum(val**2 for val in vec2.values()) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _get_search_result(
        self, query: str, keys: List[Tuple]
    ) -> List[str]:
        if not keys:
            return []

        # Update cache with current keys
        self._update_cache(keys)

        # Preprocess query
        query_preprocessed = self._preprocess_text(query)
        query_tokens = self._tokenize(query_preprocessed)
        query_tf = self._compute_tf(query_tokens)
        query_tfidf = self._compute_tfidf_vector(query_tf)

        # Compute similarities
        similarities = []
        for uid, _ in keys:
            if uid in self._tf_cache:
                doc_tfidf = self._compute_tfidf_vector(self._tf_cache[uid])
                similarity = self._cosine_similarity(query_tfidf, doc_tfidf)
                similarities.append((uid, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top fetch_num results
        top_results = similarities[: self.fetch_num]
        return [uid for uid, _ in top_results]

    def get_state_dict(self) -> dict:
        """
        Return the current state as a dictionary for merging across workers.
        """
        return {
            "text_cache": self._text_cache.copy(),
            "cached_uids": list(self._cached_uids),
            "total_docs": self._total_docs,
            "tf_cache": {
                uid: dict(counter)
                for uid, counter in self._tf_cache.items()
            },
            "df_cache": dict(self._df_cache),
            "idf_cache": self._idf_cache.copy(),
        }

    def merge_state_dict(self, state_dict: dict) -> None:
        """
        Merge another worker's state into the current state.
        This combines caches from multiple workers and recomputes global statistics.
        """
        if not state_dict:
            return

        incoming_text_cache = state_dict.get("text_cache", {})
        incoming_tf_cache = state_dict.get("tf_cache", {})

        # Merge text and TF caches (union of all cached entries)
        for uid, text in incoming_text_cache.items():
            if uid not in self._text_cache:
                self._text_cache[uid] = text
                self._cached_uids.add(uid)
                self._total_docs += 1

                # Add TF cache
                if uid in incoming_tf_cache:
                    self._tf_cache[uid] = Counter(incoming_tf_cache[uid])

                    # Update document frequency
                    for term in self._tf_cache[uid]:
                        self._df_cache[term] += 1

        # Recompute IDF based on merged statistics
        self._update_idf()

        logger.info(
            f"Merged state: now have {len(self._cached_uids)} cached entries, "
            f"{self._total_docs} total docs"
        )


class MostRecentSearcher(BaseSearcher):
    """
    Searcher that returns the most recently added entries.

    This searcher leverages Python 3.7+ dictionary insertion order guarantees.
    Since get_search_info() returns entries from memory.entries.items(), which
    maintains insertion order, we can simply take the last N entries.

    This is particularly useful when combined with:
    - FIFOUpdater: Returns the newest entries (opposite of eviction order)
    - LRUUpdater: Returns most recently added entries (not to be confused with access time)
    - Any scenario where recent information is most relevant

    Unlike RandomSearcher, this is deterministic and prioritizes temporal locality.
    Unlike similarity-based searchers, this requires no computation and is O(1).
    """

    def _get_search_result(
        self, query: str, keys: List[Tuple]
    ) -> List[str]:
        """
        Return the most recent entries based on insertion order.

        Args:
            query: Query string (ignored, kept for interface compatibility)
            keys: List of (uid, search_field) tuples in insertion order

        Returns:
            List[str]: UIDs of the most recent fetch_num entries
        """
        if not keys:
            return []

        keys_num = len(keys)

        # If we have fewer entries than fetch_num, return all
        if keys_num <= self.fetch_num:
            return [uid for uid, _ in keys]

        # Take the last fetch_num entries (most recent)
        recent_keys = keys[-self.fetch_num :]
        return [uid for uid, _ in recent_keys]
