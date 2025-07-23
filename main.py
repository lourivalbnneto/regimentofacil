import re
import os
import time
import hashlib
from io import BytesIO
import pdfplumber
import openai
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import logging
from datetime import datetime
from typing import List, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    file_url: str
    condominio_id: str

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

def sanitize_text(text: str) -> str:
    text = text.replace('\r', '').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[~:]+', '', text)
    return text.strip()

def is_valid_chunk(text: str) -> bool:
    if not text:
        return False
    if len(text) < 50:
        return False
    if re.fullmatch(r'Art\.?\s*\d+\s*\d*', text.strip()):
        return False
    if len(re.findall(r'\w+', text)) < 10:
        return False
    return True

def is_clean_text(text: str) -> bool:
    if len(re.findall(r'[~]{2,}|\w{1,2}\s+\n', text)) > 2:
        return False
    if len(re.findall(r'[^�-]+', text)) > 5:
        return False
    return True

def extract_text_from_pdf(file_url: str) -> list:
    all_text = []
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        pdf_file = BytesIO(response.content)
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    sanitized_text = sanitize_text(text)
                    all_text.append((page_number, sanitized_text))
        return all_text
    except Exception as e:
        raise Exception(f"Erro ao processar o PDF: {e}")

def chunk_by_artigos(text: str, page_number: int, parent_metadata: Dict, depth: int) -> List[Dict]:
    artigo_pattern = re.compile(r'(Art\.?\s*\d+[º°]?[^A-Z]*?)(?=Art\.?\s*\d+[º°]?|\Z)', re.IGNORECASE | re.DOTALL)
    matches = artigo_pattern.findall(text)
    chunks = []
    for match in matches:
        chunk_text = match.strip()
        if is_valid_chunk(chunk_text) and is_clean_text(chunk_text):
            current_metadata = {"type": "artigo"}
            current_metadata.update(parent_metadata)
            chunks.append({
                "page": page_number,
                "text": chunk_text,
                "metadata": current_metadata
            })
    return chunks

def split_text_into_chunks(text: str, page_number: int, parent_metadata: Dict = None, depth: int = 0) -> list:
    if depth > 2:
        return []
    chunks = []
    if parent_metadata is None:
        parent_metadata = {}
    artigo_chunks = chunk_by_artigos(text, page_number, parent_metadata, depth)
    if artigo_chunks:
        chunks.extend(artigo_chunks)
        return chunks
    return []

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def generate_chunk_hash(chunk_text: str) -> str:
    return hashlib.sha256(chunk_text.encode()).hexdigest()

def check_chunk_exists(chunk_hash: str) -> bool:
    response = supabase.table("pdf_embeddings_textos").select("id").eq("chunk_hash", chunk_hash).execute()
    return bool(response.data)

def insert_embeddings_to_supabase(chunks_with_metadata: list) -> None:
    for item in chunks_with_metadata:
        if check_chunk_exists(item["chunk_hash"]):
            continue
        try:
            supabase.table("pdf_embeddings_textos").insert(item).execute()
        except Exception as e:
            logger.error(f"Erro ao inserir chunk: {e}")

def vectorize_pdf(file_url: str, condominio_id: str) -> list:
    nome_documento = os.path.basename(file_url)
    origem = "upload_local"
    pages = extract_text_from_pdf(file_url)
    if not pages:
        return []
    all_chunks = []
    for page_number, page_text in pages:
        chunks = split_text_into_chunks(page_text, page_number)
        for chunk in chunks:
            chunk_text = chunk["text"]
            if not is_valid_chunk(chunk_text) or not is_clean_text(chunk_text):
                continue
            chunk_hash = generate_chunk_hash(chunk_text)
            try:
                embedding = get_embedding(chunk_text)
            except Exception as e:
                logger.error(f"Erro ao gerar embedding: {e}")
                embedding = None
            all_chunks.append({
                "condominio_id": condominio_id,
                "nome_documento": nome_documento,
                "origem": origem,
                "pagina": chunk.get("page", page_number),
                "chunk_text": chunk_text,
                "chunk_hash": chunk_hash,
                "embedding": embedding,
                "referencia_detectada": ""
            })
        time.sleep(0.5)
    return all_chunks

@app.get("/")
def home():
    return {"message": "FastAPI está funcionando!"}

@app.post("/vetorizar")
async def vetorizar_pdf(item: Item):
    file_url = item.file_url
    condominio_id = item.condominio_id
    nome_documento = os.path.basename(file_url)
    try:
        verifica = supabase.table("pdf_artigos_extraidos").select("id").eq("condominio_id", condominio_id).eq("nome_documento", nome_documento).execute()
        if not verifica.data:
            supabase.table("pdf_artigos_extraidos").insert({
                "condominio_id": condominio_id,
                "nome_documento": nome_documento,
                "status": "pendente"
            }).execute()
        supabase.table("pdf_embeddings_textos").delete().eq("condominio_id", condominio_id).eq("nome_documento", nome_documento).execute()
        vectorized_data = vectorize_pdf(file_url, condominio_id)
        if vectorized_data:
            insert_embeddings_to_supabase(vectorized_data)
            supabase.table("pdf_artigos_extraidos").update({
                "vetorizado": True,
                "vetorizado_em": datetime.utcnow().isoformat(),
                "status": "completo"
            }).eq("condominio_id", condominio_id).eq("nome_documento", nome_documento).execute()
            return {
                "success": True,
                "message": f"Vetorização completada com sucesso! {len(vectorized_data)} chunks processados."
            }
        else:
            raise HTTPException(status_code=400, detail="Nenhum dado foi extraído do PDF.")
    except Exception as e:
        logger.exception("vetorizar_pdf: ERRO DETALHADO")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
