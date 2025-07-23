import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from utils_pdf import extract_and_chunk_pdf
from utils_openai import gerar_embeddings_para_chunks
from utils_db import insert_chunks_into_supabase

load_dotenv()

app = FastAPI()

# CORS liberado para FlutterFlow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Altere se quiser restringir origens
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo da requisição
class VetorizacaoRequest(BaseModel):
    url_pdf: str
    nome_documento: str
    condominio_id: str
    id_usuario: str
    origem: str

@app.post("/vetorizar")
async def vetorizar(request: VetorizacaoRequest):
    try:
        logging.info(f"Iniciando vetorização do PDF: {request.url_pdf}")
        
        chunks = extract_and_chunk_pdf(
            request.url_pdf,
            nome_documento=request.nome_documento,
            condominio_id=request.condominio_id,
            id_usuario=request.id_usuario,
            origem=request.origem
        )

        if not chunks:
            return {"success": False, "message": "Nenhum chunk gerado.", "code": "chunks_vazios"}

        chunks = gerar_embeddings_para_chunks(chunks)
        insert_chunks_into_supabase(chunks)

        return {
            "success": True,
            "message": f"{len(chunks)} chunks vetorizados e salvos com sucesso.",
            "code": "vetorizacao_ok"
        }

    except Exception as e:
        logging.exception("Erro durante vetorização:")
        return {
            "success": False,
            "message": str(e),
            "code": "erro_vetorizacao"
        }
