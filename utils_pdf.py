import re
import logging
import requests
from io import BytesIO
from pdfminer.high_level import extract_text
from nltk.tokenize import sent_tokenize
import hashlib
import openai
import os

logger = logging.getLogger("utils_pdf")

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

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

def gerar_embedding(texto: str) -> list:
    try:
        response = openai.embeddings.create(
            input=[texto],
            model=EMBEDDING_MODEL,
            dimensions=1536,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"⚠️ Erro ao gerar embedding: {e}")
        return []

def gerar_hash(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()

def chunk_text_by_titles(text: str, condominio_id: str, id_usuario: str, origem: str) -> list:
    chunks = []
    chunk_base = {
        "condominio_id": condominio_id,
        "id_usuario": id_usuario,
        "origem": origem,
        "foi_vetorizada": True,
        "reusada": False,
        "acessos": 0,
        "score_similaridade": None,
        "qualidade": "Pendente",
        "pagina": None,
        "nome_documento": None,
        "referencia_detectada": None,
    }

    padrao_artigo = re.compile(r'(Art\.?[\sº°]*\d+[^\n]*)', re.IGNORECASE)
    partes = padrao_artigo.split(text)
    partes = [p.strip() for p in partes if p.strip()]

    if len(partes) < 2:
        logger.warning("⚠️ Nenhum artigo encontrado. Aplicando fallback por parágrafos.")
        paragrafos = re.split(r'(?<=\.)\s+(?=[A-ZÁÉÍÓÚ])', text)
        for paragrafo in paragrafos:
            chunk_text = paragrafo.strip()
            if len(chunk_text) < 10:
                continue
            chunk = chunk_base.copy()
            chunk["chunk_text"] = chunk_text
            chunk["chunk_hash"] = gerar_hash(chunk_text)
            chunk["embedding"] = gerar_embedding(chunk_text)
            chunks.append(chunk)
        logger.info(f"Total de chunks gerados: {len(chunks)}")
        return chunks

    for i in range(0, len(partes), 2):
        if i + 1 < len(partes):
            titulo = partes[i]
            conteudo = partes[i + 1]
            paragrafos = re.split(r'(?<=\.)\s+(?=[A-ZÁÉÍÓÚ])', conteudo)

            for paragrafo in paragrafos:
                chunk_text = f"{titulo.strip()} - {paragrafo.strip()}"
                if len(chunk_text) < 10:
                    continue
                chunk = chunk_base.copy()
                chunk["chunk_text"] = chunk_text
                chunk["chunk_hash"] = gerar_hash(chunk_text)
                chunk["embedding"] = gerar_embedding(chunk_text)
                chunks.append(chunk)

    logger.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks
