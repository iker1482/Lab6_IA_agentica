"""Piezas compartidas entre ingest.py, check_index.py y el agente: modelo de embeddings y ChromaDB.

Por defecto usa el mismo modelo local del Lab 5 (multilingual-e5-small): gratis, sin cuota.
Alternativa: EMBEDDINGS=gemini en .env usa la API de Gemini (ver el enunciado).

Prueba tu instalación (descarga el modelo la primera vez, ~490 MB):
    python -m rag_agent.rag_core
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AGENT_DIR.parent
load_dotenv(AGENT_DIR / ".env")  # la misma .env que usa `adk web`

DB_DIR = PROJECT_DIR / "chroma_db"  # ruta ABSOLUTA: adk web corre desde otro directorio
EMBEDDINGS = os.getenv("EMBEDDINGS", "local")  # "local" o "gemini"


def embedding_function():
    if EMBEDDINGS == "gemini":
        # gemini-embedding-001 regresa un vector por texto (ojo: gemini-embedding-2 NO, ver enunciado)
        return embedding_functions.GoogleGeminiEmbeddingFunction(
            model_name="gemini-embedding-001", dimension=768, api_key_env_var="GOOGLE_API_KEY")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-small",  # máx. 512 tokens (~300 palabras) por texto
        device="cpu",
        normalize_embeddings=True,
    )


def get_collection():
    """Colección persistente. Una colección por tipo de embeddings: no se pueden mezclar."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(
        name=f"corpus_{EMBEDDINGS}",
        embedding_function=embedding_function(),
        configuration={"hnsw": {"space": "cosine"}},
    )


if __name__ == "__main__":
    vec = embedding_function()(["¿Funciona el modelo de embeddings?"])[0]
    print(f"✅ Embeddings '{EMBEDDINGS}' listos: vectores de {len(vec)} dimensiones. Índice en {DB_DIR}")
