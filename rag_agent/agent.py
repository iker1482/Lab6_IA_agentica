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


def search_documents(query: str) -> dict:
    """TODO: búsqueda semántica en tu colección.

    Requisitos:
      - Expón como parámetros opcionales los filtros de metadatos que el modelo pueda
        necesitar (p. ej. doc_type, source, year…) y conviértelos en el `where` de Chroma.
      - Usa get_collection().query(query_texts=[query], n_results=..., where=...).
      - Regresa un dict con, para cada resultado, al menos el texto, el archivo y la página.
      - Si no hay resultados, regresa un dict que se lo diga claramente al modelo.
    """
    raise NotImplementedError


# TODO: al menos UNA herramienta más, de tu elección. Ideas:
#   - list_documents(): catálogo de documentos/tipos/años (para que el agente sepa qué filtros existen)
#   - read_page(source, page): el texto completo de una página (más contexto alrededor de un resultado)
#   - una herramienta propia de tu dominio


INSTRUCTION = """TODO: escribe las instrucciones del agente. Como mínimo:
- cuándo y cómo buscar (y cuándo usar filtros o varias búsquedas),
- citar cada afirmación como [archivo, p. N],
- responder "No encontré esa información en los documentos." cuando no esté en el corpus."""

root_agent = Agent(
    name="rag_agent",
    model=Gemini(
        model=AGENT_MODEL,
        # 429 = te pasaste de la cuota por minuto; 503 = modelo saturado → espera y reintenta
        retry_options=types.HttpRetryOptions(attempts=4, initial_delay=10, max_delay=60,
                                             http_status_codes=[429, 503]),
    ),
    description="Responde preguntas sobre un corpus de PDFs citando archivo y página.",
    instruction=INSTRUCTION,
    tools=[search_documents],  # TODO: agrega tus otras herramientas
)
