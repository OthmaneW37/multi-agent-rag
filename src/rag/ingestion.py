"""Module d'ingestion des documents pour le pipeline RAG."""

import os
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader, Document


def load_documents(data_path: str) -> List[Document]:
    """
    Charge tous les documents du répertoire data_path.
    
    Supports : PDF, TXT, CSV, JSON, MD, et autres formats communs
    """
    data_dir = Path(data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Le répertoire {data_path} n'existe pas.")

    try:
        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            recursive=True,
            filename_as_id=True,
        )
        documents = reader.load_data()
        
        # Ajouter les métadonnées de type si absentes
        for doc in documents:
            if "type" not in doc.metadata:
                source = doc.metadata.get("file_name", "")
                if source.endswith(".pdf"):
                    doc.metadata["type"] = "pdf"
                elif source.endswith(".csv"):
                    doc.metadata["type"] = "csv"
                elif source.endswith(".json"):
                    doc.metadata["type"] = "json"
                elif source.endswith(".txt"):
                    doc.metadata["type"] = "txt"
                else:
                    doc.metadata["type"] = "unknown"

        print(f"{len(documents)} documents chargés depuis {data_path}")
        return documents
    except Exception as e:
        print(f"Erreur lors du chargement des documents : {e}")
        return []
