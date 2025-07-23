import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils_pdf import processar_pdf_url
from utils_db import salvar_chunks_no_supabase
from utils_openai import gerar_embeddings_para_chunks

# Configurar logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Inicializar FastAPI
app = FastAPI()

# Habilitar CORS (inclusive para FlutterFlow)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pode ser substituído por seu domínio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class VetorizarRequest(BaseModel):
    url_pdf: str
    id_condominio: str
    id_usuario: str
    origem: str = "upload"

@app.post("/vetorizar")
async def vetorizar(request: VetorizarRequest):
    try:
        url_pdf = request.url_pdf
        condominio_id = request.id_condominio
        id_usuario = request.id_usuario
        origem = request.origem

        logger.info(
            f"Iniciando vetorização: url_pdf='{url_pdf}';; "
            f"id_condominio='{condominio_id}' id_usuario='{id_usuario}' origem='{origem}'"
        )

        # Extrai e processa chunks do PDF
        chunks = processar_pdf_url(url_pdf, condominio_id, id_usuario, origem)

        logger.info(f"Total de chunks gerados: {len(chunks)}")

        if not chunks:
            raise Exception("Nenhum chunk gerado")

        # Gerar embeddings
        chunks = gerar_embeddings_para_chunks(chunks)

        # Inserir no Supabase
        salvar_chunks_no_supabase(chunks)

        return {"success": True, "total_chunks": len(chunks)}

    except Exception as e:
        logger.error(f"Erro ao vetorizar PDF: {str(e)}")
        return {"success": False, "error": str(e)}
