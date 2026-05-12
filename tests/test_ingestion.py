"""Tests unitaires du module d'ingestion RAG."""

import tempfile
from pathlib import Path

import pytest

from src.rag.ingestion import load_documents


class TestIngestion:
    """Tests de la fonction load_documents."""

    def test_load_documents_txt(self):
        """Doit charger un fichier TXT et retourner un Document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "test.txt"
            f.write_text("Wydad AC vs Raja CA : 2-1", encoding="utf-8")

            docs = load_documents(tmpdir)
            assert len(docs) == 1
            assert "Wydad AC" in docs[0].text
            assert docs[0].metadata.get("type") == "txt"

    def test_load_documents_multiple(self):
        """Doit charger plusieurs fichiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.txt").write_text("Match 1")
            (Path(tmpdir) / "b.txt").write_text("Match 2")

            docs = load_documents(tmpdir)
            assert len(docs) == 2

    def test_load_documents_repertoire_inexistant(self):
        """Doit lever FileNotFoundError si le repertoire n'existe pas."""
        with pytest.raises(FileNotFoundError):
            load_documents("/chemin/inexistant/12345")

    def test_load_documents_vide(self):
        """Doit retourner une liste vide si le repertoire est vide."""
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = load_documents(tmpdir)
            assert docs == []
