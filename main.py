from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import openai
import time
import hashlib
import pdfplumber
import nltk
import re
from supabase import create_client, Client
from io import BytesIO
import requests
import logging
import sys
from datetime import datetime

app = FastAPI(root_path=os.getenv("ROOT_PATH", ""))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://regimentofacil.flutterflow.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    file_url: str
    condominio_id: str

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def limpar_texto(texto):
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(texto.split())

def extract_text_from_pdf(file_url):
    all_text = []
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception(f"Erro ao baixar o PDF: {response.status_code}")
    pdf_file = BytesIO(response.content)
    with pdfplumber.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                all_text.append((page_number, text.strip()))
            else:
                print(f"⚠️ Página {page_number} sem texto extraível.")
    return all_text

def split_article_into_chunks(article_text):
    # Divide incisos (I., 1., a), etc.)
    pattern = r'(\n\s*(?:[IVXLCDM]+\.)|\n\s*\d+\.)|\n\s*[a-z]\)'
    parts = re.split(pattern, article_text)
    chunks = []
    if len(parts) == 1:
        chunks.append(article_text.strip())
    else:
        base = parts[0].strip()
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ''
            chunks.append(f"{heading} {content}".strip())
    return chunks

def split_by_articles(text):
    pattern = r'(Art(?:igo)?\.?\s*\d+[ºo]?)'
    split_parts = re.split(pattern, text)
    articles = []
    for i in range(1, len(split_parts), 2):
        artigo_numero = split_parts[i].strip()
        artigo_texto = split_parts[i + 1].strip() if i + 1 < len(split_parts) else ''
        articles.append((artigo_numero, artigo_texto))
    return articles

def get_embedding(text, model="text-embedding-3-small"):
    if not text.strip():
        raise ValueError("Texto do chunk está vazio!")
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def generate_chunk_hash(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()

def check_chunk_exists(chunk_hash):
    response = supabase.table("pdf_embeddings_textos").select("id").eq("chunk_hash", chunk_hash).execute()
    return bool(response.data)

def insert_embeddings_to_supabase(chunks_with_metadata):
    for i, item in enumerate(chunks_with_metadata):
        if check_chunk_exists(item["chunk_hash"]):
            print(f"⚠️ Chunk {i+1} já existe. Pulando.")
            continue
        response = supabase.table("pdf_embeddings_textos").insert(item).execute()
        if response.data:
            print(f"✅ Chunk {i+1} inserido com sucesso!")
        else:
            print(f"❌ Erro ao inserir chunk {i+1}. Detalhes: {response}")

def vectorize_pdf(file_url, condominio_id):
    nome_documento = os.path.basename(file_url)
    origem = "upload_local"
    print("📄 Extraindo texto do PDF...")
    pages = extract_text_from_pdf(file_url)
    if not pages:
        print("⚠️ Nenhum texto extraído.")
        return []

    all_chunks = []
    for page_number, page_text in pages:
        print(f"✂️ Página {page_number}: dividindo por artigos...")
        articles = split_by_articles(page_text)
        print(f"🔎 Artigos detectados: {len(articles)}")
        for artigo_numero, artigo_texto in articles:
            sub_chunks = split_article_into_chunks(artigo_texto)
            for sub in sub_chunks:
                chunk = limpar_texto(f"{artigo_numero} {sub}".strip())
                if not chunk:
                    continue
                chunk_hash = generate_chunk_hash(chunk)
                try:
                    embedding = get_embedding(chunk)
                except Exception as e:
                    print(f"❌ Erro ao gerar embedding: {e}")
                    embedding = None
                all_chunks.append({
                    "condominio_id": condominio_id,
                    "nome_documento": nome_documento,
                    "origem": origem,
                    "pagina": page_number,
                    "chunk_text": chunk,
                    "chunk_hash": chunk_hash,
                    "embedding": embedding
                })
                time.sleep(0.5)
    return all_chunks

@app.post("/vetorizar")
async def vetorizar_pdf(item: Item):
    try:
        file_url = item.file_url
        condominio_id = item.condominio_id
        if not file_url or not condominio_id:
            return {"error": "Parâmetros obrigatórios ausentes"}, 400

        nome_documento = os.path.basename(file_url)
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
            return {"success": True, "message": f"Vetorização completada com sucesso! {len(vectorized_data)} chunks processados."}
        else:
            return {"error": "Nenhum dado foi extraído do PDF."}, 400
    except Exception as e:
        return {"error": f"Erro interno: {str(e)}"}, 500

if __name__ == "__main__":
    print("🔍 Verificando conexões com serviços externos...")
    openai_ok = openai.api_key is not None
    supabase_ok = SUPABASE_URL and SUPABASE_KEY
    if openai_ok and supabase_ok:
        print("\n✅ Todos os serviços estão funcionando corretamente!")
        sys.exit(0)
    else:
        print("\n❌ Há problemas com alguns serviços. Verifique os erros acima.")
        sys.exit(1)
