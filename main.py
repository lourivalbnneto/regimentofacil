from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from utils_pdf import extract_and_chunk_pdf
from utils_openai import gerar_embeddings_para_chunks
from utils_db import inserir_chunks_supabase
import logging
import os

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Middleware CORS para permitir chamadas de outras origens (como FlutterFlow)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Altere para origens específicas em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vetorizar")
async def vetorizar_pdf(request: Request):
    try:
        body = await request.json()
        url_pdf = body.get("url_pdf")
        condominio_id = body.get("condominio_id")
        id_usuario = body.get("id_usuario")
        nome_documento = body.get("nome_documento")
        origem = body.get("origem", "upload")

        if not all([url_pdf, condominio_id, id_usuario, nome_documento]):
            return {"success": False, "message": "Parâmetros obrigatórios ausentes."}

        chunks = extract_and_chunk_pdf(
            url_pdf=url_pdf,
            nome_documento=nome_documento,
            condominio_id=condominio_id,
            id_usuario=id_usuario,
            origem=origem
        )

        chunks = await gerar_embeddings_para_chunks(chunks)
        inseridos = inserir_chunks_supabase(chunks)

        return {
            "success": True,
            "message": f"{len(inseridos)} chunks inseridos com sucesso.",
            "chunks_inseridos": len(inseridos)
        }

    except Exception as e:
        logging.exception("Erro ao processar a vetorização do PDF")
        return {"success": False, "message": str(e)}
