import re
import os
import time
import hashlib
from io import BytesIO
import pdfplumber
import openai
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import logging
from datetime import datetime

# Configuração do FastAPI
app = FastAPI(root_path=os.getenv("ROOT_PATH", ""))

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://regimentofacil.flutterflow.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração do Modelo
class Item(BaseModel):
    file_url: str
    condominio_id: str

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carregamento das variáveis de ambiente
load_dotenv()

# Configuração das chaves da OpenAI e Supabase
openai.api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Conexão com o Supabase
supabase: Client = create_client(supabase_url, supabase_key)

# Funções Utilitárias
def sanitize_text(text: str) -> str:
    """Limpa o texto removendo caracteres de nova linha e espaços extras."""
    text = text.replace('\r', ' ').replace('\n', ' ')
    return ' '.join(text.strip().split())

def extract_text_from_pdf(file_url: str) -> list:
    """Extrai o texto de todas as páginas de um PDF."""

    all_text = []
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Garante que a requisição foi bem-sucedida
        pdf_file = BytesIO(response.content)
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    all_text.append((page_number, sanitize_text(text)))
                else:
                    logger.warning(f"Página {page_number} sem texto extraível.")
        return all_text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Erro ao baixar o PDF: {e}")
    except pdfplumber.PDFOpeningError as e:
        raise Exception(f"Erro ao abrir o PDF: {e}")

def extract_references(text: str) -> list:
    """Extrai referências do texto (Art., §, Inciso, alínea)."""

    references = []

    # Regex para Artigo (Art. 1º, Art. 1, Artigo 1)
    match = re.search(r'\b(Art(?:igo)?\.?\s*\d+[º°]?)', text, re.IGNORECASE)
    if match:
        references.append(match.group(1).strip())

    # Regex para Parágrafo (§ 1º, Parágrafo único)
    match = re.search(r'\b(§\s*\d+[º°]?|Parágrafo(?:\s+único|\s+primeiro|segundo|terceiro)?)', text, re.IGNORECASE)
    if match:
        references.append(match.group(1).strip())

    # Regex para Inciso (I, II, III...)
    match = re.search(r'\b([IVXLCDM]+)[).]', text)
    if match:
        references.append(f"Inciso {match.group(1)}")

    # Regex para Alínea (a, b, c...)
    match = re.search(r'\b([a-z])[).]', text)
    if match:
        references.append(f"Alínea {match.group(1)}")

    return list(dict.fromkeys(references))  # Remove duplicatas mantendo a ordem

def split_text_into_chunks(text: str) -> list:
    """Divide o texto em chunks menores, identificando seções por marcadores."""

    # Regex simplificada para marcadores (Art., §, Inciso, Alínea)
    pattern = re.compile(r'''
        (?=
            \s*
            (?:
                Art(?:igo)?\.?\s*\d+[º°]?|
                §+\s*\d+[º°]?|
                Parágrafo(?:\s+único|\s+primeiro|segundo|terceiro)|
                [IVXLCDM]+[).]|
                [a-z][).])
        )
    ''', re.VERBOSE | re.IGNORECASE)

    chunks = []
    matches = list(pattern.finditer(text))

    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        chunk_text = sanitize_text(chunk_text)

        if len(chunk_text) < 15:
            continue

        references = extract_references(chunk_text)
        chunks.append({
            "text": chunk_text,
            "references": references
        })

    return chunks

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Gera o embedding para o texto usando o modelo da OpenAI."""

    if not text.strip():
        raise ValueError("Texto do chunk está vazio!")
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def generate_chunk_hash(chunk_text: str) -> str:
    """Gera um hash para o texto do chunk."""

    return hashlib.sha256(chunk_text.encode()).hexdigest()

def check_chunk_exists(chunk_hash: str) -> bool:
    """Verifica se um chunk com o hash especificado já existe no banco de dados."""

    response = supabase.table("pdf_embeddings_textos").select("id").eq("chunk_hash", chunk_hash).execute()
    return bool(response.data)

def insert_embeddings_to_supabase(chunks_with_metadata: list) -> None:
    """Insere os embeddings e metadados dos chunks no Supabase."""

    for i, item in enumerate(chunks_with_metadata):
        if check_chunk_exists(item["chunk_hash"]):
            logger.warning(f"Chunk {i+1} já existe. Pulando.")
            continue

        response = supabase.table("pdf_embeddings_textos").insert(item).execute()
        if response.data:
            logger.info(f"Chunk {i+1} inserido com sucesso!")
        else:
            logger.error(f"Erro ao inserir chunk {i+1}. Detalhes: {response.error}")

def vectorize_pdf(file_url: str, condominio_id: str) -> list:
    """Processa o PDF, extrai o texto, divide em chunks, gera embeddings e salva no Supabase."""

    nome_documento = os.path.basename(file_url)
    origem = "upload_local"
    logger.info("Extraindo texto do PDF...")
    pages = extract_text_from_pdf(file_url)

    if not pages:
        logger.warning("Nenhum texto extraído.")
        return []

    all_chunks = []
    for page_number, page_text in pages:
        logger.info(f"Página {page_number}: extraindo por marcadores...")
        chunks = split_text_into_chunks(page_text)
        logger.info(f"Chunks detectados: {len(chunks)}")

        for chunk in chunks:
            chunk_text = chunk["text"]
            references = chunk["references"]
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
                "pagina": page_number,
                "chunk_text": chunk_text,
                "chunk_hash": chunk_hash,
                "embedding": embedding,
                "referencia_detectada": " | ".join(references)
            })
        time.sleep(0.5)  # Adiciona um pequeno delay entre o processamento de cada página
    return all_chunks

# Rotas da API
@app.get("/")
def home():
    return {"message": "FastAPI está funcionando!"}

@app.post("/vetorizar")
async def vetorizar_pdf_endpoint(item: Item):
    """Endpoint para iniciar a vetorização de um PDF."""

    file_url = item.file_url
    condominio_id = item.condominio_id

    if not file_url or not condominio_id:
        logger.error("Parâmetros obrigatórios ausentes")
        return {"error": "Parâmetros 'file_url' e 'condominio_id' são obrigatórios."}, 400

    nome_documento = os.path.basename(file_url)

    # Verifica se já existe um registro para este condomínio e documento
    verifica = supabase.table("pdf_artigos_extraidos").select("id").eq("condominio_id", condominio_id).eq("nome_documento", nome_documento).execute()
    if not verifica.data:
        logger.info("Inserindo novo registro em pdf_artigos_extraidos")
        supabase.table("pdf_artigos_extraidos").insert({
            "condominio_id": condominio_id,
            "nome_documento": nome_documento,
            "status": "pendente"
        }).execute()

    # Deleta os registros existentes para este condomínio e documento
    supabase.table("pdf_embeddings_textos").delete().eq("condominio_id", condominio_id).eq("nome_documento", nome_documento).execute()

    logger.info(f"Iniciando processamento do PDF: {file_url}")
    vectorized_data = vectorize_pdf(file_url, condominio_id)

    if vectorized_data:
        logger.info(f"Inserindo {len(vectorized_data)} chunks no Supabase")
        insert_embeddings_to_supabase(vectorized_data)

        # Atualiza o status do processamento
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
    print("Verificando conexões com serviços externos...")
    # Você pode adicionar aqui verificações para Supabase e OpenAI
    # e usar sys.exit(1) em caso de falha
