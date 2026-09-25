"""Ingesta: PDFs en data/ → chunks con metadatos → embeddings → ChromaDB persistente.

Uso:
    python ingest.py --dry-run      # extrae y genera chunks SIN calcular embeddings → revisa chunks_preview.jsonl
    python ingest.py                # indexa en ChromaDB
    python ingest.py --reset        # borra la colección y reindexa todo (p. ej. si cambiaste el chunking, o modelo de embedding)

Tú decides:
  - cómo extraer el texto (pymupdf4llm, pypdf, …) y cómo limpiarlo,
  - cómo crear los chunks (por página, por encabezado, por tamaño, con overlap...),
  - qué metadatos guardar (mínimo: source, page, doc_type + uno propio de tu dominio).

Lo que YA está hecho en rag_agent/rag_core.py (léelo antes de empezar):
  - get_collection(): colección persistente de ChromaDB en ./chroma_db, con distancia coseno y
    el modelo de embeddings ya configurado (multilingual-e5-small, igual que en el Lab 5).
    Chroma calcula los embeddings solo: collection.add(documents=...) y collection.query(query_texts=...).
"""

import argparse
import json
from pathlib import Path

from rag_agent.rag_core import PROJECT_DIR, get_collection

DATA_DIR = PROJECT_DIR / "data"
BATCH = 50  # chunks por collection.add


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extrae el texto de un PDF, una entrada por página: [{"page": 1, "text": "..."}, ...].

    Problemas típicos de los PDFs que debes resolver (o justificar por qué no importan):
      - líneas cortas: cada renglón visual es una línea → los párrafos quedan partidos,
      - encabezados y pies de página que se repiten en cada hoja,
      - como hacer chunking de tablas
    """
    raise NotImplementedError


def build_chunks(pdf_path: Path) -> list[dict]:
    """Convierte un PDF en una lista de chunks listos para indexar.

    Cada chunk debe ser: {"id": str único y DETERMINISTA,
                          "text": str,
                          "metadata": {"source": str, "page": int, "doc_type": str, ...}}

    Pistas:
      - multilingual-e5-small solo lee los primeros 512 tokens (~300 palabras): lo que sobre
        de un chunk más largo se ignora sin avisar.
      - un id determinista (p. ej. archivo-página-número) permite reanudar la indexación sin duplicar.
      - los valores de metadata no pueden ser None (usa "" o 0).
    """
    raise NotImplementedError
 

def index(chunks: list[dict], collection) -> None:
    """Agrega los chunks a la colección (Chroma calcula los embeddings).

    Requisitos:
      - agrega por lotes de BATCH chunks y muestra el avance (indexar tarda unos minutos),
      - omite los chunks cuyo id ya está en la colección, para que volver a correr el script
        después de un error continúe donde se quedó.
    Usa collection.add(ids=..., documents=..., metadatas=...).
    """
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no calcula embeddings")
    ap.add_argument("--reset", action="store_true", help="borra la colección antes de indexar")
    args = ap.parse_args()

    pdfs = sorted(DATA_DIR.rglob("*.pdf"))
    chunks = [c for pdf in pdfs for c in build_chunks(pdf)]
    print(f"{len(chunks)} chunks de {len(pdfs)} PDFs")

    if args.dry_run:
        with (PROJECT_DIR / "chunks_preview.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print("Dry run: revisa chunks_preview.jsonl. No se calcularon embeddings.")
        return

    if args.reset:
        collection = get_collection()
        ids = collection.get(include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        print("Colección vaciada.")
    index(chunks, get_collection())


if __name__ == "__main__":
    main()
