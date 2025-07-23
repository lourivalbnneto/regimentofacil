from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import hashlib
import logging
import openai
import os
from utils_pdf import extract_text_from_pdf, sanitize_text, chunk_text_by_titles
from utils_db import insert_chunks_into_supabase

# Configurações
openai.api_key = os.getenv("OPENAI_API_KEY")
MAX_EMBEDDING_TOKENS = 8191

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Modelos de entrada
class PDFContent(BaseModel):
    id_condominio: str
    pdf_text: str

# Função principal de vetorização
@app.post("/vectorize_pdf")
async def vectorize_pdf(data: PDFContent):
    logger.info("Iniciando vetorização para condomínio: %s", data.id_condominio)

    try:
        # 1. Sanitiza texto completo
        texto_limpo = sanitize_text(data.pdf_text)
        if not texto_limpo:
            raise ValueError("Texto sanitizado está vazio.")

        # 2. Geração de chunks hierárquicos
        chunks = chunk_text_by_titles(texto_limpo, depth=0)
        if not chunks:
            raise ValueError("Nenhum chunk foi gerado após o chunking.")

        vetorizados = []

        for idx, chunk in enumerate(chunks):
            texto = chunk.strip()

            # Validação antes da vetorização
            if len(texto) < 20:
                logger.warning("Chunk %d ignorado: texto muito curto", idx)
                continue
            if len(texto) > 20000:
                logger.warning("Chunk %d ignorado: texto muito longo", idx)
                continue

            try:
                # 3. Embedding
                embedding = get_embedding(texto)

                # 4. Hash
                hash_id = hashlib.sha256((data.id_condominio + texto).encode("utf-8")).hexdigest()

                vetorizados.append({
                    "id_condominio": data.id_condominio,
                    "conteudo": texto,
                    "vetor": embedding,
                    "hash": hash_id
                })
            except Exception as e:
                logger.exception("Erro ao vetorizar chunk %d: %s", idx, e)
                continue

        if not vetorizados:
            raise Exception("Nenhum chunk foi vetorizado com sucesso.")

        # 5. Armazena no Supabase
        insert_chunks_into_supabase(vetorizados)
        logger.info("Vetorização concluída com sucesso. Total: %d chunks", len(vetorizados))
        return {"success": True, "message": f"{len(vetorizados)} chunks vetorizados com sucesso."}

    except Exception as e:
        logger.exception("Erro durante a vetorização geral: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# Embedding
def get_embedding(texto: str) -> List[float]:
    response = openai.Embedding.create(
        input=[texto],
        model="text-embedding-3-small"
    )
    return response["data"][0]["embedding"]
