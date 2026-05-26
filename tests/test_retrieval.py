"""Tests for retrieval layer — dense, sparse, and hybrid."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.retrieval.sparse_retrieval import SparseRetriever, _tokenize
from app.retrieval.hybrid_retrieval import _rrf_score


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_docs(texts):
    return [Document(page_content=t, metadata={"source": "test.txt"}) for t in texts]


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def test_tokenize_basic():
    tokens = _tokenize("Hello World! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []


# ── BM25 Sparse Retrieval ──────────────────────────────────────────────────────

class TestSparseRetriever:
    def setup_method(self):
        docs = make_docs([
            "machine learning is a subset of artificial intelligence",
            "python is a popular programming language for data science",
            "neural networks are inspired by the human brain",
            "retrieval augmented generation improves LLM accuracy",
        ])
        self.retriever = SparseRetriever(docs)

    def test_retrieve_returns_docs(self):
        results = self.retriever.retrieve("machine learning", k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_retrieve_empty_index(self):
        retriever = SparseRetriever([])
        results = retriever.retrieve("query", k=5)
        assert results == []

    def test_retrieve_top_result_relevant(self):
        results = self.retriever.retrieve("neural networks brain", k=3)
        assert len(results) > 0
        # Top result should be about neural networks
        assert "neural" in results[0].page_content.lower()

    def test_scores_normalized(self):
        results = self.retriever.retrieve("language python", k=4)
        for doc in results:
            assert 0.0 <= doc.metadata["sparse_score"] <= 1.0

    def test_retrieval_type_metadata(self):
        results = self.retriever.retrieve("AI", k=2)
        for doc in results:
            assert doc.metadata["retrieval_type"] == "sparse"


# ── RRF Score ─────────────────────────────────────────────────────────────────

def test_rrf_score_decreases_with_rank():
    s0 = _rrf_score(0)
    s1 = _rrf_score(1)
    s10 = _rrf_score(10)
    assert s0 > s1 > s10


def test_rrf_score_positive():
    assert _rrf_score(0) > 0
    assert _rrf_score(100) > 0


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class TestHybridRetriever:
    def test_rrf_merges_results(self):
        from app.retrieval.hybrid_retrieval import HybridRetriever

        docs = make_docs([
            "apple banana cherry",
            "deep learning transformers",
            "python data science pandas",
        ])

        # Mock dense retriever
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [
            Document(page_content="apple banana cherry",
                     metadata={"dense_score": 0.9, "source": "a"}),
            Document(page_content="python data science pandas",
                     metadata={"dense_score": 0.7, "source": "b"}),
        ]

        # Mock sparse retriever
        sparse_mock = MagicMock()
        sparse_mock.retrieve.return_value = [
            Document(page_content="deep learning transformers",
                     metadata={"sparse_score": 0.85, "source": "c"}),
            Document(page_content="apple banana cherry",
                     metadata={"sparse_score": 0.6, "source": "a"}),
        ]

        hybrid = HybridRetriever(dense_mock, sparse_mock)
        results = hybrid.retrieve(["test query"], k=5)

        assert len(results) > 0
        # apple banana cherry appears in both → should rank high
        contents = [r.page_content for r in results]
        assert "apple banana cherry" in contents

    def test_hybrid_deduplicates(self):
        from app.retrieval.hybrid_retrieval import HybridRetriever

        same_doc = Document(page_content="same content here", metadata={"source": "x"})

        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [same_doc, same_doc]

        sparse_mock = MagicMock()
        sparse_mock.retrieve.return_value = [same_doc]

        hybrid = HybridRetriever(dense_mock, sparse_mock)
        results = hybrid.retrieve(["query"], k=10)

        # Should only appear once
        contents = [r.page_content for r in results]
        assert contents.count("same content here") == 1
