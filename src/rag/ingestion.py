"""Module d'ingestion des documents pour le pipeline RAG."""

from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader, Document


def load_documents(data_path: str) -> List[Document]:
    """
    Charge tous les documents du repertoire data_path.

    Supports : PDF, TXT, CSV, JSON, MD

    Args:
        data_path: Chemin vers le repertoire contenant les documents

    Returns:
        Liste de documents LlamaIndex
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Le repertoire {data_path} n'existe pas.")

    try:
        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            recursive=True,
            filename_as_id=True,
        )
        documents = reader.load_data()

        # Ajouter les metadata de type si absentes
        for doc in documents:
            if "type" not in doc.metadata:
                source = doc.metadata.get("file_name", "")
                suffix = Path(source).suffix.lower()
                doc.metadata["type"] = suffix.lstrip(".") or "txt"

        print(f"[RAG] {len(documents)} documents charges depuis {data_path}")
        return documents
    except Exception as e:
        print(f"[RAG] Erreur lors du chargement : {e}")
        return []
