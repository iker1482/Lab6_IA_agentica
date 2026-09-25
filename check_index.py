"""Verifica que el índice esté listo para el agente.

Uso:
    python check_index.py
    python check_index.py --query "¿cuál es la vigencia del contrato?"   # prueba una búsqueda
"""

import argparse
from collections import Counter

from rag_agent.rag_core import DB_DIR, EMBEDDINGS, get_collection

REQUIRED = {"source": str, "page": int, "doc_type": str}


def check(desc, ok, hint=""):
    print(f"✅ {desc}" if ok else f"❌ {desc} — {hint}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="prueba una búsqueda")
    args = ap.parse_args()

    print(f"Índice en: {DB_DIR} · embeddings: {EMBEDDINGS}\n")
    col = get_collection()
    n = col.count()
    if not check(f"la colección tiene chunks ({n})", n > 0, "corre python ingest.py"):
        return
    data = col.get(include=["metadatas", "documents", "embeddings"], limit=n)
    metas, docs = data["metadatas"], data["documents"]

    for key, typ in REQUIRED.items():
        check(f"todos los chunks tienen '{key}' ({typ.__name__})",
              all(isinstance(m.get(key), typ) for m in metas), f"agrega '{key}' a los metadatos")
    extra = set(metas[0]) - set(REQUIRED) - {"chunk", "title", "section"}
    check(f"hay al menos un metadato propio de tu dominio {sorted(extra) or ''}", bool(extra),
          "agrega uno (año, empresa, artículo, fecha…)")
    check("las páginas empiezan en 1", min(m["page"] for m in metas) >= 1, "usa números de página humanos")
    check(f"cada chunk tiene embedding ({len(data['embeddings'][0])} dimensiones)",
          len(data["embeddings"]) == n, "vuelve a correr ingest.py")

    words = [len(d.split()) for d in docs]
    check(f"tamaño de chunk razonable (prom. {sum(words) // n} palabras, máx. {max(words)})",
          40 <= sum(words) / n <= 300, "revisa tu función de chunking")
    long_chunks = sum(w > 300 for w in words)
    check(f"ningún chunk rebasa ~300 palabras ({long_chunks} lo hacen)", long_chunks == 0,
          "multilingual-e5-small solo lee 512 tokens: el resto del chunk se ignora")
    check(f"no hay textos duplicados ({n - len(set(docs))} duplicados)", len(set(docs)) >= 0.95 * n,
          "¿encabezados/pies de página repetidos o PDFs duplicados?")

    by_type, by_src = Counter(m["doc_type"] for m in metas), Counter(m["source"] for m in metas)
    print(f"\nTipos de documento: {dict(by_type)}")
    print("Chunks por documento:")
    for src, c in by_src.most_common():
        print(f"  {c:5d}  {src}")

    if args.query:
        from rag_agent.agent import search_documents
        print(f"\nBúsqueda de prueba: {args.query!r}")
        for r in search_documents(args.query, k=3).get("results", []):
            print(f"  [{r['source']}, p. {r['page']}] d={r['distance']}  {r['text'][:120]!r}")


if __name__ == "__main__":
    main()
