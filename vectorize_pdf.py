from fastapi import FastAPI
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
import threading

# Inicializa FastAPI com suporte a root_path
app = FastAPI(root_path=os.getenv("ROOT_PATH", ""))

# Modelo de entrada
class Item(BaseModel):
    file_url: str
    condominio_id: str

# Carrega .env se existir
load_dotenv()

# Configura chaves
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Cria cliente do Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Garantir que o tokenizer esteja disponível
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Limpeza de texto
def limpar_texto(texto):
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(texto.split())

# Extrair texto do PDF
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
                print(f"⚠️ Página {page_number} sem texto.")
    return all_text

# Dividir por artigos
def split_by_articles(text):
    pattern = r'(Art(?:igo)?\.?\s*\d+[ºo]?)'
    parts = re.split(pattern, text)
    articles = []
    for i in range(1, len(parts), 2):
        numero = parts[i].strip()
        conteudo = parts[i + 1].strip() if i + 1 < len(parts) else ''
        articles.append(f"{numero} {conteudo}")
    return articles

# Gerar embedding
def get_embedding(text, model="text-embedding-3-small"):
    if not text.strip():
        raise ValueError("Texto vazio!")
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

# Hash do chunk
def generate_chunk_hash(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()

# Verifica duplicação
def check_chunk_exists(chunk_hash):
    response = supabase.table("pdf_embeddings_textos").select("id").eq("chunk_hash", chunk_hash).execute()
    return bool(response.data)

# Insere embeddings
def insert_embeddings_to_supabase(chunks):
    for i, item in enumerate(chunks):
        if check_chunk_exists(item["chunk_hash"]):
            print(f"⚠️ Chunk {i+1} já existe. Pulando.")
            continue
        response = supabase.table("pdf_embeddings_textos").insert(item).execute()
        if response.data:
            print(f"✅ Chunk {i+1} inserido!")
        else:
            print(f"❌ Erro ao inserir chunk {i+1}: {response}")

# Processamento principal
def vectorize_pdf(file_url, condominio_id):
    nome_documento = os.path.basename(file_url)
    origem = "upload_local"
    print("📄 Extraindo texto...")
    pages = extract_text_from_pdf(file_url)
    if not pages:
        return []

    all_chunks = []
    for page_number, page_text in pages:
        articles = split_by_articles(page_text)
        for article in articles:
            chunk = limpar_texto(article)
            if not chunk:
                continue
            chunk_hash = generate_chunk_hash(chunk)
            try:
                embedding = get_embedding(chunk)
            except Exception as e:
                print(f"❌ Erro embedding: {e}")
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

# Rota de teste
@app.get("/")
def home():
    return {"message": "FastAPI está funcionando!"}

# Rota de vetorização
@app.post("/vetorizar")
async def vetorizar_pdf(item: Item):
    try:
        if not item.file_url or not item.condominio_id:
            return {"error": "Parâmetros obrigatórios."}, 400
        data = vectorize_pdf(item.file_url, item.condominio_id)
        if data:
            insert_embeddings_to_supabase(data)
            return {"message": "Vetorização completada com sucesso!"}, 200
        else:
            return {"error": "Nenhum texto encontrado."}, 500
    except Exception as e:
        return {"error": str(e)}, 500

# Thread para manter ativo no Railway
def keep_alive():
    while True:
        print("🟢 App está rodando...")
        time.sleep(10)

threading.Thread(target=keep_alive, daemon=True).start()
