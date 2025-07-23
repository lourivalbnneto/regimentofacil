import requests
from io import BytesIO
from pdfminer.high_level import extract_text
import hashlib
import logging

logger = logging.getLogger("utils_pdf")

def extract_text_from_pdf(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        texto = extract_text(BytesIO(response.content))
        logger.info(f"Texto extraído (tamanho={len(texto)}): {texto[:200]}...")
        return texto
    except Exception as e:
        logger.error(f"Erro inesperado ao extrair texto do PDF: {e}")
        return ""

def sanitize_text(text):
    return text.replace("\xa0", " ").strip()

def gerar_hash(conteudo, condominio_id, id_usuario, pagina):
    base = f"{conteudo}|{condominio_id}|{id_usuario}|{pagina}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def chunk_text_by_titles(texto, condominio_id, id_usuario, origem):
    chunks = []

    try:
        if "Art. " not in texto and "Artigo" not in texto:
            logger.warning("⚠️ Nenhum artigo encontrado. Aplicando fallback por parágrafos.")
            paragrafos = [p.strip() for p in texto.split("\n") if p.strip()]
            for i, paragrafo in enumerate(paragrafos):
                chunk_hash = gerar_hash(paragrafo, condominio_id, id_usuario, i)
                chunks.append({
                    "condominio_id": condominio_id,
                    "id_usuario": id_usuario,
                    "chunk_text": paragrafo,
                    "chunk_hash": chunk_hash,
                    "pagina": i,
                    "origem": origem,
                    "foi_vetorizada": False,
                    "reusada": False,
                    "acessos": 0,
                })
        else:
            # Aqui você pode implementar chunking por artigo/subartigo futuramente
            pass

    except Exception as e:
        logger.error(f"Erro ao gerar chunks: {e}", exc_info=True)

    logger.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks
