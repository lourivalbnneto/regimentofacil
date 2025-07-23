import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from utils_pdf import extrair_chunks_pdf_por_paragrafo
from utils_db import salvar_chunks_no_supabase
from utils_openai import gerar_embeddings_para_chunks

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Carregar variáveis de ambiente
load_dotenv()

# Criar app FastAPI
app = FastAPI()

# CORS liberado para qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vetorizar")
async def vetorizar(request: Request):
    dados = await request.json()

    url_pdf = dados.get("url_pdf")
    id_condominio = dados.get("id_condominio")
    id_usuario = dados.get("id_usuario")
    origem = dados.get("origem", "upload")
    nome_documento = dados.get("nome_documento", "documento.pdf")

    logger.info(f"Iniciando vetorização: url_pdf='{url_pdf}';;; id_condominio='{id_condominio}' id_usuario='{id_usuario}' origem='{origem}'")

    try:
        # Extrair e chunkar o PDF por parágrafos
        chunks = extrair_chunks_pdf_por_paragrafo(
            url_pdf=url_pdf,
            condominio_id=id_condominio,
            id_usuario=id_usuario,
            origem=origem,
            nome_documento=nome_documento
        )
        logger.info(f"Total de chunks gerados: {len(chunks)}")

        if not chunks:
            raise Exception("Nenhum chunk gerado. Verifique o conteúdo do PDF.")

        # Gerar embeddings
        chunks_com_embeddings = gerar_embeddings_para_chunks(chunks)

        # Salvar no Supabase
        salvar_chunks_no_supabase(chunks_com_embeddings)

        return {"success": True, "chunks_processados": len(chunks)}

    except Exception as e:
        logger.error(f"Erro ao vetorizar PDF: {str(e)}")
        return {"success": False, "erro": str(e)}
