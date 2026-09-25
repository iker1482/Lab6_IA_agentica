"""Tu agente de RAG agéntico. Corre con `adk web` desde la carpeta que CONTIENE a rag_agent/.

El agente debe decidir por sí mismo cuándo buscar, qué consulta escribir, qué filtros usar
y cuántas búsquedas hacer. Tu trabajo: darle buenas herramientas y buenas instrucciones.

Recuerda: ADK usa el nombre, los parámetros (con sus tipos) y el docstring de cada función
para explicarle al modelo cuándo y cómo usarla. Un docstring pobre = un agente que decide mal.
"""

import os

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types

from .rag_core import get_collection

AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-3.5-flash-lite")

DOC_TYPES = ("partido", "temporada", "referencia")


def _build_where(**filters) -> dict | None:
    """Convierte filtros no vacíos en un `where` de Chroma (None, {k: v} o {"$and": [...]})."""
    conds = [{k: v} for k, v in filters.items() if v not in ("", None, 0)]
    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


# ---------- Herramienta 1: búsqueda ----------
def search_documents(
    query: str,
    doc_type: str = "",
    source: str = "",
    n_results: int = 5,
) -> dict:
    """Busca fragmentos relevantes en la documentación de la API de StatsBomb.

    Úsala para cualquier pregunta sobre campos, eventos, estadísticas o endpoints.
    Escribe la consulta en inglés y con los términos técnicos que aparecerían en la
    documentación (p. ej. "obv_for_after shot event" en vez de "qué tan buena fue la jugada").

    Args:
        query: texto a buscar.
        doc_type: opcional. Filtra por tipo de documento: "partido" (matches, lineups,
            events, 360 frames, match stats), "temporada" (season stats) o "referencia"
            (competitions, player mapping). Vacío = todo el corpus.
        source: opcional. Filtra por archivo exacto, p. ej. "events_v11.pdf".
            Usa list_documents() si no sabes el nombre exacto.
        n_results: cuántos fragmentos regresar (1-10). Default 5.

    Returns:
        {"query", "filters", "results": [{"text", "source", "page", "doc_type", "score"}, ...]}
        o {"results": [], "message": "..."} si no hubo coincidencias.
    """
    n_results = max(1, min(int(n_results), 10))
    where = _build_where(doc_type=doc_type.strip(), source=source.strip())
    collection = get_collection()

    if collection.count() == 0:
        return {"results": [], "message": "El índice está vacío. Corre `python ingest.py` primero."}

    res = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    if not docs:
        return {
            "results": [],
            "message": "Sin coincidencias con esos filtros. Prueba sin filtro, con otro doc_type, "
                       "o revisa el nombre del archivo con list_documents().",
        }

    results = []
    for text, meta, dist in zip(docs, metas, dists):
        results.append({
            "text": text,
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "doc_type": meta.get("doc_type", ""),
            "score": round(1 - dist, 3),  # similitud coseno: más alto = más relevante
        })

    return {"query": query, "filters": where or {}, "results": results}


# ---------- Herramienta 2: catálogo ----------
def list_documents() -> dict:
    """Lista los documentos del corpus con su tipo, versión de la API y número de páginas.

    Úsala ANTES de buscar cuando necesites saber qué archivos o tipos existen para
    elegir filtros (p. ej. al comparar dos documentos o cuando el usuario menciona
    un documento sin decir el nombre exacto). Basta con llamarla una vez por conversación.

    Returns:
        {"doc_types": [...], "documents": [{"source", "doc_type", "version", "pages", "chunks"}, ...]}
    """
    collection = get_collection()
    data = collection.get(include=["metadatas"])

    catalog: dict[str, dict] = {}
    for meta in data["metadatas"]:
        src = meta.get("source", "")
        entry = catalog.setdefault(src, {
            "source": src,
            "doc_type": meta.get("doc_type", ""),
            "version": meta.get("version", ""),
            "pages": 0,
            "chunks": 0,
        })
        entry["pages"] = max(entry["pages"], int(meta.get("page", 0)))
        entry["chunks"] += 1

    documents = sorted(catalog.values(), key=lambda d: (d["doc_type"], d["source"]))
    if not documents:
        return {"doc_types": [], "documents": [], "message": "El índice está vacío."}

    return {
        "doc_types": sorted({d["doc_type"] for d in documents}),
        "documents": documents,
    }


# ---------- Herramienta 3: página completa ----------
def read_page(source: str, page: int) -> dict:
    """Regresa todo el texto indexado de una página específica de un documento.

    Úsala cuando un resultado de search_documents está cortado (p. ej. una tabla de
    campos que continúa en el siguiente fragmento) y necesitas el contexto completo
    antes de responder. No la uses para explorar: primero busca con search_documents.

    Args:
        source: nombre exacto del archivo, p. ej. "events_v11.pdf".
        page: número de página (empieza en 1).

    Returns:
        {"source", "page", "text"} o {"text": "", "message": "..."} si la página no existe.
    """
    collection = get_collection()
    data = collection.get(
        where={"$and": [{"source": source.strip()}, {"page": int(page)}]},
        include=["documents", "metadatas"],
    )
    if not data["documents"]:
        return {"source": source, "page": page, "text": "",
                "message": "No hay texto indexado para esa página. Verifica el nombre con list_documents()."}

    # ordena los chunks por id (los ids deterministas archivo-página-n conservan el orden)
    pairs = sorted(zip(data["ids"], data["documents"]))
    text = "\n\n".join(doc for _, doc in pairs)
    return {"source": source, "page": page, "text": text}


INSTRUCTION = """Eres un asistente experto en la documentación de la API de StatsBomb
(datos de fútbol: partidos, alineaciones, eventos, frames 360, estadísticas de jugador
y de equipo por partido y por temporada, competiciones y mapeo de jugadores).
Respondes SOLO con lo que está en los documentos indexados.

## Cuándo buscar
- Saludos, agradecimientos o charla casual: responde directo, SIN usar herramientas.
- Cualquier pregunta técnica: busca con search_documents ANTES de responder.
  Nunca contestes con tu conocimiento general aunque creas saber la respuesta.
- No repitas una búsqueda con la misma consulta y los mismos filtros. Si la primera
  no sirvió, cambia la consulta (otros términos, en inglés) o los filtros.
- Escribe las consultas en inglés y con los nombres técnicos de los campos.

## Filtros y varias búsquedas
- Si la pregunta menciona un documento o nivel concreto ("por temporada", "en events",
  "de lineups"), usa el filtro doc_type o source correspondiente.
- Si la pregunta COMPARA dos o más documentos (p. ej. stats por partido vs por temporada),
  haz UNA búsqueda por cada documento con su filtro `source`, y luego compara.
- Si no sabes qué archivos existen o el usuario no dice el nombre exacto, llama primero
  a list_documents y usa el nombre que regresa.
- Si un fragmento está cortado (una tabla que sigue), usa read_page con ese source y page.

## Citas
- Cita CADA afirmación con el archivo y la página del fragmento de donde salió,
  con el formato exacto [archivo, p. N]. Ejemplo: [events_v11.pdf, p. 12].
- Si una respuesta junta varios fragmentos, cita cada uno donde corresponde.
- Nunca inventes una cita ni cites una página que no viste en los resultados.

## Cuando no está en los documentos
- Si después de buscar (y de intentar al menos una consulta alternativa cuando tenga
  sentido) no encuentras la información, responde exactamente:
  "No encontré esa información en los documentos."
  No agregues una respuesta de tu conocimiento general después de esa frase.

## Formato
- Responde en el idioma del usuario (normalmente español), de forma breve y precisa.
- Los nombres de campos, eventos y endpoints déjalos tal cual aparecen en la documentación.
"""

root_agent = Agent(
    name="rag_agent",
    model=Gemini(
        model=AGENT_MODEL,
        # 429 = te pasaste de la cuota por minuto; 503 = modelo saturado → espera y reintenta
        retry_options=types.HttpRetryOptions(attempts=4, initial_delay=10, max_delay=60,
                                             http_status_codes=[429, 503]),
    ),
    description="Responde preguntas sobre la documentación de la API de StatsBomb citando archivo y página.",
    instruction=INSTRUCTION,
    tools=[search_documents, list_documents, read_page],
)
