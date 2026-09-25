"""build_chunks.py

Recorre una carpeta de PDFs con documentación técnica (formato tipo StatsBomb API docs:
título + secciones con encabezados + tablas de especificación + bloques JSON de ejemplo)
y genera UN chunk por archivo, preservando título, tablas y texto de contexto.

Uso:
    python build_chunks.py --input-dir data/ --output-file chunks.jsonl
    python build_chunks.py --input-dir data/ --output-file chunks.jsonl --log-file build_chunks.log

Cada línea del .jsonl de salida es un chunk:
    {
      "chunk_id": "chunk_1",
      "source_file": "API 360 Frames v2.0.0.pdf",
      "title": "...",
      "tables": [{"headers": [...], "rows": [[...], ...]}, ...],
      "text": "...",
      "raw_markdown": "..."
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("build_chunks")


# --------------------------------------------------------------------------- #
# Extracción de markdown (con fallback a pypdf si pymupdf4llm falla)
# --------------------------------------------------------------------------- #

def extract_markdown(pdf_path: Path) -> str:
    """Devuelve el contenido del PDF en Markdown.

    Intenta pymupdf4llm (preserva encabezados y tablas nativamente). Si falla
    (PDF corrupto, dependencia rota, etc.) cae a pypdf, extrayendo texto plano
    por página y devolviéndolo como markdown "pobre" (sin tablas estructuradas),
    para que el resto del pipeline no se detenga.
    """
    try:
        import pymupdf4llm

        md = pymupdf4llm.to_markdown(str(pdf_path))
        if not md or not md.strip():
            raise ValueError("pymupdf4llm devolvió contenido vacío")
        return md
    except Exception as e:  # noqa: BLE001 - queremos capturar cualquier fallo de extracción
        logger.warning(
            "pymupdf4llm falló en %s (%s). Usando fallback con pypdf (sin tablas).",
            pdf_path.name,
            e,
        )
        return _extract_with_pypdf(pdf_path)


def _extract_with_pypdf(pdf_path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:  # noqa: BLE001
        logger.error("No se pudo abrir %s con pypdf: %s", pdf_path.name, e)
        return ""

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("Página %d de %s sin texto extraíble: %s", i + 1, pdf_path.name, e)
            text = ""
        if text.strip():
            pages_text.append(text)
        else:
            logger.warning("Página %d de %s vino vacía.", i + 1, pdf_path.name)

    return "\n\n".join(pages_text)


# --------------------------------------------------------------------------- #
# Parsing de markdown: título, tablas, texto de contexto
# --------------------------------------------------------------------------- #

_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def extract_title(markdown: str, fallback: str) -> str:
    """Primera línea no vacía y no-tabla del markdown, limpiando símbolos '#'."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _TABLE_ROW_RE.match(stripped):
            continue
        return re.sub(r"^#+\s*", "", stripped).strip()
    logger.warning("No se encontró título extraíble; usando nombre de archivo como fallback.")
    return fallback


def _split_row(line: str) -> list[str]:
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") else inner
    return [cell.strip() for cell in inner.split("|")]


def extract_tables_and_text(markdown: str) -> tuple[list[dict], str]:
    """Recorre el markdown línea por línea, extrayendo tablas y dejando el resto como texto.

    Reconoce el patrón estándar de tabla markdown:
        | Header1 | Header2 | ...
        | ------- | ------- | ...
        | val1    | val2    | ...
    Filas mal formadas (número de celdas distinto al header) se registran como
    advertencia y se omiten, sin detener el proceso.
    """
    lines = markdown.splitlines()
    tables: list[dict] = []
    text_lines: list[str] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        is_header = _TABLE_ROW_RE.match(line)
        is_next_sep = (i + 1 < n) and _TABLE_SEP_RE.match(lines[i + 1] or "") and "|" in (lines[i + 1] or "")

        if is_header and is_next_sep:
            headers = _split_row(line)
            rows: list[list[str]] = []
            j = i + 2
            while j < n and _TABLE_ROW_RE.match(lines[j]):
                row = _split_row(lines[j])
                if len(row) != len(headers):
                    logger.warning(
                        "Fila de tabla mal formada (esperaba %d columnas, encontró %d): %r",
                        len(headers),
                        len(row),
                        lines[j],
                    )
                else:
                    rows.append(row)
                j += 1
            tables.append({"headers": headers, "rows": rows})
            i = j
            continue

        text_lines.append(line)
        i += 1

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(text_lines)).strip()
    return tables, text


# --------------------------------------------------------------------------- #
# Construcción de un chunk por PDF
# --------------------------------------------------------------------------- #

def build_chunk(pdf_path: Path, chunk_index: int) -> dict:
    logger.info("Procesando %s -> chunk_%d", pdf_path.name, chunk_index)
    markdown = extract_markdown(pdf_path)

    if not markdown.strip():
        logger.error("Sin contenido extraíble en %s; se genera chunk vacío.", pdf_path.name)

    title = extract_title(markdown, fallback=pdf_path.stem)
    tables, text = extract_tables_and_text(markdown)

    return {
        "chunk_id": f"chunk_{chunk_index}",
        "source_file": pdf_path.name,
        "title": title,
        "tables": tables,
        "text": text,
        "raw_markdown": markdown,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def setup_logging(log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="data", help="Carpeta con los PDFs (default: data/)")
    ap.add_argument("--output-file", default="chunks.jsonl", help="Archivo .jsonl de salida")
    ap.add_argument("--log-file", default=None, help="Archivo opcional para guardar el log")
    args = ap.parse_args()

    setup_logging(args.log_file)

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists():
        logger.error("La carpeta de entrada no existe: %s", input_dir)
        raise SystemExit(1)

    pdfs = sorted(input_dir.rglob("*.pdf"))
    if not pdfs:
        logger.warning("No se encontraron PDFs en %s", input_dir)

    chunks = []
    for idx, pdf_path in enumerate(pdfs, start=1):
        try:
            chunk = build_chunk(pdf_path, idx)
            chunks.append(chunk)
        except Exception as e:  # noqa: BLE001 - un PDF corrupto no debe detener el resto
            logger.error("Fallo irrecuperable procesando %s: %s", pdf_path.name, e)
            continue

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info("%d chunks escritos en %s (de %d PDFs)", len(chunks), output_file, len(pdfs))


if __name__ == "__main__":
    main()
