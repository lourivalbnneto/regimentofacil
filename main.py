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

def split_by_paragraphs_with_referencia(text):
    pattern = r'(?=(Art(?:igo)?\.?\s*\d+[ºo]?|[IVXLCDM]{1,5}[.)]|[a-z]{1}[)]))'
    raw_chunks = re.split(pattern, text)
    final_chunks = []

    current_referencia = ""
    buffer = ""

    for i in range(1, len(raw_chunks), 2):
        marcador = raw_chunks[i].strip()
        trecho = raw_chunks[i + 1].strip() if i + 1 < len(raw_chunks) else ""

        # Detecta tipo da referência
        if re.match(r'Art', marcador, re.IGNORECASE):
            current_referencia = marcador
        elif re.match(r'[IVXLCDM]{1,5}[.)]', marcador):
            current_referencia += f" | Inciso {marcador}"
        elif re.match(r'[a-z]{1}[)]', marcador):
            current_referencia += f" | alínea {marcador}"

        paragrafo = f"{marcador} {trecho}".strip()
        paragrafo = limpar_texto(paragrafo)

        if paragrafo:
            final_chunks.append({
                "referencia": current_referencia.strip(" |"),
                "texto": paragrafo
            })

    return final_chunks

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
        print(f"✂️ Página {page_number}: dividindo por parágrafos e referências...")
        chunks = split_by_paragraphs_with_referencia(page_text)
        print(f"🔎 Chunks detectados: {len(chunks)}")

        for chunk_obj in chunks:
            referencia = chunk_obj["referencia"]
            texto_base = chunk_obj["texto"]
            texto_completo = f"{referencia}: {texto_base}" if referencia else texto_base

            chunk_hash = generate_chunk_hash(texto_completo)
            try:
                embedding = get_embedding(texto_completo)
            except Exception as e:
                print(f"❌ Erro ao gerar embedding: {e}")
                embedding = None

            all_chunks.append({
                "condominio_id": condominio_id,
                "nome_documento": nome_documento,
                "origem": origem,
                "pagina": page_number,
                "chunk_text": texto_completo,
                "chunk_hash": chunk_hash,
                "embedding": embedding,
                "referencia_detectada": referencia
            })
            time.sleep(0.5)
    return all_chunks

def check_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY não encontrada no arquivo .env")
        return False
    try:
        openai.api_key = api_key
        response = openai.embeddings.create(model="text-embedding-3-small", input="Teste de conexão")
        print("✅ Conexão com a OpenAI estabelecida com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro na conexão com a OpenAI: {str(e)}")
        return False

def check_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("❌ SUPABASE_URL ou SUPABASE_KEY não encontradas no arquivo .env")
        return False
    try:
        supabase = create_client(url, key)
        response = supabase.table("pdf_embeddings_textos").select("count", count="exact").limit(1).execute()
        count = response.count
        print(f"✅ Conexão com o Supabase estabelecida com sucesso! ({count} registros na tabela)")
        return True
    except Exception as e:
        print(f"❌ Erro na conexão com o Supabase: {str(e)}")
        return False

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
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao baixar o PDF: {str(e)}")
        return {"error": f"Erro ao baixar o PDF: {str(e)}"}, 500
    except openai.OpenAIError as e:
        logger.error(f"Erro na API da OpenAI: {str(e)}")
        return {"error": f"Erro na API da OpenAI: {str(e)}"}, 500
    except Exception as e:
        logger.exception(f"Erro não esperado: {str(e)}")
        return {"error": f"Erro interno: {str(e)}"}, 500

if __name__ == "__main__":
    print("🔍 Verificando conexões com serviços externos...")
    openai_ok = check_openai()
    supabase_ok = check_supabase()

    if openai_ok and supabase_ok:
        print("\n✅ Todos os serviços estão funcionando corretamente!")
        sys.exit(0)
    else:
        print("\n❌ Há problemas com alguns serviços. Verifique os erros acima.")
        sys.exit(1)
