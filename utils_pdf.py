import re
import logging
import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    sanitized = re.sub(r'\s+', ' ', text).strip()
    logger.debug(f"Texto sanitizado (tamanho={len(sanitized)}): {sanitized[:100]}...")
    return sanitized


def chunk_text_by_titles(text: str, condominio_id: str, id_usuario: str, origem: str) -> list:
    chunks = []
    chunk_base = {
        "condominio_id": condominio_id,
        "id_usuario": id_usuario,
        "origem": origem,
        "foi_vetorizada": False,
        "reusada": False,
        "acessos": 0,
        "score_similaridade": None,
        "qualidade": "Pendente",
    }

    padrao_artigo = re.compile(r'(Art\.?[\sº°]*\d+[^\n]*)', re.IGNORECASE)
    partes = padrao_artigo.split(text)
    partes = [p.strip() for p in partes if p.strip()]

    if not partes or len(partes) < 2:
        logger.warning("⚠️ Nenhum artigo encontrado. Aplicando fallback por parágrafos.")
        paragrafos = re.split(r'(?<=\.)\s+(?=[A-ZÁÉÍÓÚ])', text)
        for p in paragrafos:
            chunk = chunk_base.copy()
            chunk["chunk_text"] = p.strip()
            chunks.append(chunk)
    else:
        for i in range(0, len(partes), 2):
            if i + 1 < len(partes):
                titulo = partes[i]
                conteudo = partes[i + 1]
                paragrafos = re.split(r'(?<=\.)\s+(?=[A-ZÁÉÍÓÚ])', conteudo)
                for p in paragrafos:
                    chunk = chunk_base.copy()
                    chunk["chunk_text"] = f"{titulo.strip()} - {p.strip()}"
                    chunks.append(chunk)

    logger.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks
