import re
from typing import List, Dict
import logging
from pdfminer.high_level import extract_text
import requests
from io import BytesIO
from nltk.tokenize import sent_tokenize

# Função principal para extrair texto e gerar chunks
def extract_and_chunk_pdf(url_pdf: str, nome_documento: str, condominio_id: str, id_usuario: str, origem: str) -> List[Dict]:
    response = requests.get(url_pdf)
    response.raise_for_status()

    texto_completo = extract_text(BytesIO(response.content))
    logging.info(f"Texto extraído (tamanho={len(texto_completo)}): {texto_completo[:300]}")

    chunks = chunk_text_by_articles(
        texto_completo,
        nome_documento=nome_documento,
        condominio_id=condominio_id,
        id_usuario=id_usuario,
        origem=origem
    )
    return chunks

# Sanitização básica
def clean_text(text: str) -> str:
    text = re.sub(r'[^\S\r\n]+', ' ', text)  # espaços múltiplos
    text = re.sub(r'\s*\n\s*', '\n', text)  # quebras de linha
    return text.strip()

# Dividir por artigos (Art. 1º, Artigo 2º, etc.)
def chunk_text_by_articles(text: str, nome_documento: str, condominio_id: str, id_usuario: str, origem: str) -> List[Dict]:
    text = clean_text(text)
    regex_artigo = r'(Art\.?[\sº°]*\d+[A-Za-zº°]*)\s*[-–:]?\s*'

    partes = re.split(regex_artigo, text)
    if len(partes) <= 1:
        logging.warning("⚠️ Nenhum artigo encontrado. Aplicando fallback por parágrafos.")
        return fallback_por_paragrafo(text, nome_documento, condominio_id, id_usuario, origem)

    chunks = []
    pagina = 1
    for i in range(1, len(partes), 2):
        referencia = partes[i].strip()
        conteudo = partes[i + 1].strip()
        parags = [p for p in conteudo.split('\n') if p.strip()]

        for par in parags:
            frases = sent_tokenize(par.strip(), language='portuguese')
            if not frases:
                continue
            chunk = {
                "condominio_id": condominio_id,
                "id_usuario": id_usuario,
                "nome_documento": nome_documento,
                "pagina": pagina,
                "chunk_text": " ".join(frases),
                "chunk_hash": "",
                "embedding": None,
                "qualidade": "pendente",
                "referencia_detectada": f"Art. {referencia}",
                "origem": origem,
                "foi_vetorizada": False,
                "reusada": False,
                "score_similaridade": None
            }
            chunks.append(chunk)

    logging.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks

# Caso não detecte "Art." aplicar fallback por parágrafos
def fallback_por_paragrafo(text: str, nome_documento: str, condominio_id: str, id_usuario: str, origem: str) -> List[Dict]:
    text = clean_text(text)
    parags = [p for p in text.split('\n') if p.strip()]
    pagina = 1
    chunks = []

    for par in parags:
        frases = sent_tokenize(par.strip(), language='portuguese')
        if not frases:
            continue
        chunk = {
            "condominio_id": condominio_id,
            "id_usuario": id_usuario,
            "nome_documento": nome_documento,
            "pagina": pagina,
            "chunk_text": " ".join(frases),
            "chunk_hash": "",
            "embedding": None,
            "qualidade": "pendente",
            "referencia_detectada": None,
            "origem": origem,
            "foi_vetorizada": False,
            "reusada": False,
            "score_similaridade": None
        }
        chunks.append(chunk)

    logging.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks
