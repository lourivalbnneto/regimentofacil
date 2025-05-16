from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import openai
import time
import hashlib
import pdfplumber
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
def limpar_texto(texto):
    texto = texto.replace('\r', ' ').replace('\n', ' ')
    return ' '.join(texto.strip().split())

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

def extrair_referencia(paragrafo):
    """Extrai a principal referência do parágrafo, se houver."""
    referencia = []

    match_artigo = re.search(r'\b(Art(?:igo)?\.?\s*\d+[º°o]?)', paragrafo, re.IGNORECASE)
    if match_artigo:
        referencia.append(match_artigo.group(1).strip())

    match_paragrafo = re.search(r'\b(§\s*\d+[º°o]?|Parágrafo(?:\s+único|\s+primeiro|s*segundo|s*terceiro)?)', paragrafo, re.IGNORECASE)
    if match_paragrafo:
        referencia.append(match_paragrafo.group(1).strip())

    match_inciso = re.search(r'\b([IVXLCDM]+)[.)-]', paragrafo)
    if match_inciso:
        referencia.append(f"Inciso {match_inciso.group(1)}")

    match_alinea = re.search(r'\b([a-zA-Z])[.)-]', paragrafo)
    if match_alinea:
        referencia.append(f"alínea {match_alinea.group(1)})")

    match_decimal = re.search(r'\b(\d{2,3}\.\d+)', paragrafo)
    if match_decimal:
        referencia.append(f"Item {match_decimal.group(1)}")

    return ' | '.join(dict.fromkeys(referencia))  # remove duplicatas mantendo ordem
def extrair_chunks_com_referencias(texto):
    """Divide texto em chunks por parágrafo e extrai referência de cada um."""
    linhas = texto.split('\n')
    paragrafos = []
    buffer = ""

    for linha in linhas:
        linha = linha.strip()
        if linha == "":
            if buffer:
                paragrafos.append(buffer.strip())
                buffer = ""
        else:
            buffer += " " + linha
    if buffer:
        paragrafos.append(buffer.strip())

    chunks = []
    for p in paragrafos:
        p_limpo = limpar_texto(p)
        if not p_limpo:
            continue
        ref = extrair_referencia(p_limpo)
        chunk_text = f"{ref}: {p_limpo}" if ref else p_limpo
        chunks.append({
            "texto": chunk_text,
            "referencia": ref
        })

    return chunks
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
        print(f"✂️ Página {page_number}: extraindo parágrafos...")
        chunks = extrair_chunks_com_referencias(page_text)
        print(f"🔎 Parágrafos detectados: {len(chunks)}")

        for chunk_obj in chunks:
            chunk_text = chunk_obj["texto"]
            referencia = chunk_obj["referencia"]
            chunk_hash = generate_chunk_hash(chunk_text)
            try:
                embedding = get_embedding(chunk_text)
            except Exception as e:
                print(f"❌ Erro ao gerar embedding: {e}")
                embedding = None

            all_chunks.append({
                "condominio_id": condominio_id,
                "nome_documento": nome_documento,
                "origem": origem,
                "pagina": page_number,
                "chunk_text": chunk_text,
                "chunk_hash": chunk_hash,
                "embedding": embedding,
                "referencia_detectada": referencia
            })
            time.sleep(0.5)
    return all_chunks
@app.get("/")
def home():
    return {"message": "FastAPI está funcionando!"}

@app.post("/vetorizar")
async def vetorizar_pdf(item: Item):
    try:
        file_url = item.file_url
        condominio_id = item.condominio_id

        if not file_url or not condominio_id:
            logger.error("Parâmetros obrigatórios ausentes")
            return {"error": "Parâmetros 'file_url' e 'condominio_id' são obrigatórios."}, 400

        nome_documento = os.path.basename(file_url)

        verifica = supabase.table("pdf_artigos_extraidos").select("id").eq("condominio_id", condominio_id).eq("nome_documento", nome_documento).execute()
        if not verifica.data:
            logger.info("Inserindo novo registro em pdf_artigos_extraidos")
            supabase.table("pdf_artigos_extraidos").insert({
                "condominio_id": condominio_id,
                "nome_documento": nome_documento,
                "status": "pendente"
            }).execute()

        supabase.table("pdf_embeddings_textos").delete().eq("condominio_id", condominio_id).eq("nome_documento", nome_documento).execute()

        logger.info(f"Iniciando processamento do PDF: {file_url}")
        vectorized_data = vectorize_pdf(file_url, condominio_id)

        if vectorized_data:
            logger.info(f"Inserindo {len(vectorized_data)} chunks no Supabase")
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
            logger.warning("Nenhum dado foi vetorizado")
            return {"error": "Nenhum dado foi extraído do PDF."}, 400
    except Exception as e:
        logger.exception(f"Erro não esperado: {str(e)}")
        return {"error": f"Erro interno: {str(e)}"}, 500

if __name__ == "__main__":
    print("🔍 Verificando conexões com serviços externos...")
    sys.exit(0)
