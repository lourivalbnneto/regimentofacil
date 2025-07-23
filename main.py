# main.py com logger.exception detalhado

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
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
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

def _chunk_by_titles_recursive(text: str, page_number: int, parent_metadata: Dict) -> List[Dict]:
    title_pattern = re.compile(
        r'(?:(CAP[IÍ]TULO\s+[IVXLCDM]+)|(SE[CÇ][AÃ]O\s+[A-Z]+)|(Art\.\s*\d+[º°]?))\s*(.*?)(?=\n|$)',
        re.IGNORECASE
    )
    matches = list(title_pattern.finditer(text))
    if not matches:
        return []
    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        title = matches[i].group(0).strip()
        current_metadata = {"type": "title", "title": title}
        current_metadata.update(parent_metadata)
        if len(chunk_text) > 300:
            sub_chunks = split_text_into_chunks(chunk_text, page_number, current_metadata)
            chunks.extend(sub_chunks)
        elif is_valid_chunk(chunk_text):
            chunks.append({
                "page": page_number,
                "text": chunk_text,
                "metadata": current_metadata
            })
    return chunks

def _chunk_by_paragraphs(text: str, page_number: int, parent_metadata: Dict) -> List[Dict]:
    chunks = []
    paragraphs = text.split('\n\n')
    for paragraph in paragraphs:
        chunk_text = paragraph.strip()
        if is_valid_chunk(chunk_text):
            current_metadata = {"type": "paragraph"}
            current_metadata.update(parent_metadata)
            chunks.append({
                "page": page_number,
                "text": chunk_text,
                "metadata": current_metadata
            })
    return chunks

def _chunk_by_sentences(text: str, page_number: int, parent_metadata: Dict) -> List[Dict]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 350:
            current_chunk += sentence + " "
        else:
            chunk_text = current_chunk.strip()
            if is_valid_chunk(chunk_text):
                current_metadata = {"type": "sentence"}
                current_metadata.update(parent_metadata)
                chunks.append({
                    "page": page_number,
                    "text": chunk_text,
                    "metadata": current_metadata
                })
            current_chunk = sentence + " "
    if current_chunk:
        chunk_text = current_chunk.strip()
        if is_valid_chunk(chunk_text):
            current_metadata = {"type": "sentence"}
            current_metadata.update(parent_metadata)
            chunks.append({
                "page": page_number,
                "text": chunk_text,
                "metadata": current_metadata
            })
    return chunks

def split_text_into_chunks(text: str, page_number: int, parent_metadata: Dict = None) -> list:
    chunks = []
    if parent_metadata is None:
        parent_metadata = {}
    title_chunks = _chunk_by_titles_recursive(text, page_number, parent_metadata)
    if title_chunks:
        chunks.extend(title_chunks)
        return chunks
    para_chunks = _chunk_by_paragraphs(text, page_number, parent_metadata)
    if para_chunks:
        chunks.extend(para_chunks)
        return chunks
    sentence_chunks = _chunk_by_sentences(text, page_number, parent_metadata)
    chunks.extend(sentence_chunks)
    return chunks

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
            if not is_valid_chunk(chunk_text):
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