from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback
import logging
from utils_pdf import extract_text_from_pdf, sanitize_text, chunk_text_by_titles

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Altere para o domínio do seu app em produção, se quiser restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar logging
logging.basicConfig(level=logging.INFO)

class VetorizarRequest(BaseModel):
    url_pdf: str
    id_condominio: str
    id_usuario: str
    origem: Optional[str] = "upload"

@app.post("/vetorizar")
async def vetorizar(request: VetorizarRequest):
    try:
        logging.info(f"Iniciando vetorização: {request}")
        texto_extraido = extract_text_from_pdf(request.url_pdf)
        texto_limpo = sanitize_text(texto_extraido)
        chunks = chunk_text_by_titles(texto_limpo)
        logging.info(f"Total de chunks gerados: {len(chunks)}")

        # Aqui você pode salvar os chunks no Supabase ou outro destino
        return {"success": True, "message": "Vetorização concluída", "total_chunks": len(chunks)}

    except Exception as e:
        logging.error(f"Erro ao vetorizar PDF: {e}")
        logging.error(traceback.format_exc())
        return {"success": False, "message": "Erro interno ao processar PDF", "error": str(e)}
