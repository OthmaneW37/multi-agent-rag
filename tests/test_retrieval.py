"""Tests unitaires du module de retrieval RAG."""

from unittest.mock import MagicMock

import pytest

from src.rag.retrieval import retrieve_context, query_index


class MockNode:
    """Mock d'un Node LlamaIndex."""

    def __init__(self, text, metadata, score=0.5):
        self.text = text
        self.metadata = metadata
        self.score = score


class MockRetrieverResult:
    """Mock d'un resultat de retriever."""

    def __init__(self, node, score=0.5):
        self.node = node
        self.score = score


class TestRetrieveContext:
    """Tests de retrieve_context."""

    def test_retrieve_avec_resultats(self, monkeypatch):
        """Doit retourner les resultats filtres par score."""
        mock_index = MagicMock()
        mock_retriever = MagicMock()

        node1 = MockNode("Wydad gagne 2-1", {"source": "fixtures.txt"}, score=0.35)
        node2 = MockNode("Raja perd", {"source": "fixtures.txt"}, score=0.10)

        mock_retriever.retrieve.return_value = [
            MockRetrieverResult(node1, score=0.35),
            MockRetrieverResult(node2, score=0.10),
        ]

        # Monkeypatch create_retriever
        monkeypatch.setattr(
            "src.rag.retrieval.create_retriever",
            lambda index, similarity_top_k: mock_retriever,
        )

        results = retrieve_context(mock_index, "forme Wydad", top_k=5, min_score=0.15)
        assert len(results) == 1
        assert results[0]["text"] == "Wydad gagne 2-1"
        assert results[0]["score"] == 0.35

    def test_retrieve_sans_resultats(self, monkeypatch):
        """Doit retourner une liste vide si aucun resultat ne depasse le seuil."""
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        monkeypatch.setattr(
            "src.rag.retrieval.create_retriever",
            lambda index, similarity_top_k: mock_retriever,
        )

        results = retrieve_context(mock_index, "inexistant", top_k=5)
        assert results == []


class TestQueryIndex:
    """Tests de query_index."""

    def test_query_avec_resultats(self, monkeypatch):
        """Doit retourner les passages concatenes."""
        mock_index = MagicMock()

        def fake_retrieve(index, query, top_k, min_score):
            return [
                {"text": "Stat 1", "score": 0.3, "metadata": {}, "source": "a.txt"},
            ]

        monkeypatch.setattr("src.rag.retrieval.retrieve_context", fake_retrieve)

        reponse = query_index(mock_index, "question test")
        assert "Stat 1" in reponse
        assert "a.txt" in reponse

    def test_query_sans_resultats(self, monkeypatch):
        """Doit retourner un message d'absence de donnees."""
        mock_index = MagicMock()

        def fake_retrieve(index, query, top_k, min_score):
            return []

        monkeypatch.setattr("src.rag.retrieval.retrieve_context", fake_retrieve)

        reponse = query_index(mock_index, "inexistant")
        assert "Aucun document" in reponse
