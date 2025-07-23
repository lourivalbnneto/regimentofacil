import os
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils_pdf import extract_and_chunk_pdf
from utils_db import salvar_chunks_no_supabase
from utils_openai import gerar_embeddings_para_chunks

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

class VetorizacaoRequest(BaseModel):
    url_pdf: str
    id_condominio: str
    id_usuario: str

@app.post("/vetorizar")
async def vetorizar(request: VetorizacaoRequest):
    try:
        url_pdf = request.url_pdf
        condominio_id = request.id_condominio
        id_usuario = request.id_usuario
        origem = "upload"
        nome_documento = url_pdf.split("/")[-1]  # nome do arquivo PDF

        logger.info(f"Iniciando vetorização: url_pdf='{url_pdf}'; id_condominio='{condominio_id}' id_usuario='{id_usuario}' origem='{origem}'")

        chunks = extract_and_chunk_pdf(url_pdf, nome_documento, condominio_id, id_usuario, origem)
        logger.info(f"Total de chunks gerados: {len(chunks)}")

        if not chunks:
            return {"success": False, "message": "Nenhum chunk gerado", "status": 204}

        chunks_com_embeddings = gerar_embeddings_para_chunks(chunks)
        salvar_chunks_no_supabase(chunks_com_embeddings)

        return {"success": True, "message": "Chunks vetorizados com sucesso", "status": 200, "quantidade_chunks": len(chunks)}

    except Exception as e:
        logger.error(f"Erro ao vetorizar PDF: {str(e)}", exc_info=True)
        return {"success": False, "message": str(e), "status": 500}
