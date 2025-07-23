import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from utils_pdf import extract_text_from_pdf, sanitize_text, chunk_text_by_titles
from utils_db import salvar_chunks_no_supabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS liberado para uso no FlutterFlow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/vetorizar")
async def vetorizar(request: Request):
    try:
        payload = await request.json()
        url_pdf = payload["url_pdf"]
        id_condominio = payload["id_condominio"]
        id_usuario = payload["id_usuario"]
        origem = payload.get("origem", "upload")

        logger.info(
            f"Iniciando vetorização: url_pdf='{url_pdf}'; id_condominio='{id_condominio}' id_usuario='{id_usuario}' origem='{origem}'"
        )

        texto_extraido = extract_text_from_pdf(url_pdf)
        texto_limpo = sanitize_text(texto_extraido)
        chunks = chunk_text_by_titles(texto_limpo, id_condominio, id_usuario, origem)

        logger.info(f"Total de chunks gerados: {len(chunks)}")

        salvar_chunks_no_supabase(chunks)
        return {"success": True, "message": "Vetorização concluída", "total_chunks": len(chunks)}

    except Exception as e:
        logger.error(f"Erro ao vetorizar PDF: {e}", exc_info=True)
        return {"success": False, "message": "Erro na vetorização", "error": str(e)}
