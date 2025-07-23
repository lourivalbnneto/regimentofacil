import os
import logging
import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE = "pdf_embeddings_textos"

def salvar_chunks_no_supabase(chunks):
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase URL ou Key não definida")
        raise Exception("Variáveis de ambiente do Supabase não definidas")

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    for chunk in chunks:
        response = requests.post(url, headers=headers, json=chunk)
        if not response.ok:
            logger.error("Erro ao inserir chunk no Supabase: %s", response.text)
            raise Exception(f"Erro ao inserir chunk: {response.text}")