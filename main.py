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
import logging
import sys

# Iniciar aplicação FastAPI
app = FastAPI(root_path=os.getenv("ROOT_PATH", ""))

# Modelo de entrada para a rota POST
class Item(BaseModel):
    file_url: str
    condominio_id: str

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

# Configurações de API
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Cliente Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Garantir que o tokenizer do NLTK esteja disponível
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Limpar quebras de linha e espaços múltiplos
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
                print(f"⚠️ Página {page_number} sem texto extraível.")
    return all_text

# Separar por artigos
def split_by_articles(text):
    pattern = r'(Art(?:igo)?\.?\s*\d+[ºo]?)'
    split_parts = re.split(pattern, text)

    articles = []
    for i in range(1, len(split_parts), 2):
        artigo_numero = split_parts[i].strip()
        artigo_texto = split_parts[i + 1].strip() if i + 1 < len(split_parts) else ''
        articles.append(f"{artigo_numero} {artigo_texto}")
    return articles

# Gerar embedding
def get_embedding(text, model="text-embedding-3-small"):
    if not text.strip():
        raise ValueError("Texto do chunk está vazio!")
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

# Gerar hash
def generate_chunk_hash(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()

# Verificar duplicidade
def check_chunk_exists(chunk_hash):
    response = supabase.table("pdf_embeddings_textos").select("id").eq("chunk_hash", chunk_hash).execute()
    return bool(response.data)

# Inserir no Supabase
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

# Processamento completo
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

        for article in articles:
            chunk = limpar_texto(article.strip())
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

# Verificar conexão com OpenAI
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

# Verificar conexão com Supabase
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

# Rota raiz para teste
@app.get("/")
def home():
    return {"message": "FastAPI está funcionando!"}

# Rota POST para vetorização
@app.post("/vetorizar")
async def vetorizar_pdf(item: Item):
    try:
        file_url = item.file_url
        condominio_id = item.condominio_id

        if not file_url or not condominio_id:
            logger.error("Parâmetros obrigatórios ausentes")
            return {"error": "Parâmetros 'file_url' e 'condominio_id' são obrigatórios."}, 400

        logger.info(f"Iniciando processamento do PDF: {file_url}")
        vectorized_data = vectorize_pdf(file_url, condominio_id)

        if vectorized_data:
            logger.info(f"Inserindo {len(vectorized_data)} chunks no Supabase")
            insert_embeddings_to_supabase(vectorized_data)
            return {"message": f"Vetorização completada com sucesso! {len(vectorized_data)} chunks processados."}
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

# Ponto de entrada para execução direta
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