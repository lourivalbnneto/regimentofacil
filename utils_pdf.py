import logging
from io import BytesIO
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

import requests
from io import BytesIO
from pdfminer.high_level import extract_text
import logging

import re
import unicodedata

logger = logging.getLogger(__name__)


def extract_text_from_pdf(url_pdf: str) -> str:
    try:
        response = requests.get(url_pdf)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        return extract_text(pdf_file)
    except Exception as e:
        logger.error(f"Erro inesperado ao extrair texto do PDF: {e}")
        return ""


def sanitize_text(text: str) -> str:
    # Remove caracteres invisíveis e normaliza
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.replace("\x00", "").replace("\u200b", "")
    return text.strip()


def chunk_text_by_titles(text: str) -> list[dict]:
    """
    Divide o texto por artigos e parágrafos.
    """
    pattern = r"(Art\.?\s*\d+[º°]?(?:-[A-Z])?(?:\s*–)?(?:\s*\(.*?\))?)"
    partes = re.split(pattern, text, flags=re.IGNORECASE)

    chunks = []
    for i in range(1, len(partes), 2):
        titulo = partes[i].strip()
        conteudo = partes[i + 1].strip()
        chunk = f"{titulo} - {conteudo}"
        chunks.append({
            "title": titulo,
            "content": chunk
        })

    return chunks
