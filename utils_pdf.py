import re
import logging

logger = logging.getLogger(__name__)

def sanitize_text(text):
    if not text:
        return ""
    return re.sub(r"[\s\u200b]+", " ", text).strip()

def chunk_text_by_titles(texto, depth=0, max_depth=5):
    if depth > max_depth:
        logger.warning("Limite de profundidade de chunking excedido.")
        return [texto.strip()]

    padrao_artigo = re.compile(r"(?=Art\.\s*\d+º?\s*-?)", re.IGNORECASE)
    partes = padrao_artigo.split(texto)
    artigos = []

    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue
        if not parte.lower().startswith("art"):
            if artigos:
                artigos[-1] += " " + parte
            else:
                artigos.append(parte)
        else:
            artigos.append(parte)

    return artigos