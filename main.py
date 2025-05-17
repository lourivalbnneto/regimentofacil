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
from typing import List, Dict, Tuple

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
    level=logging.DEBUG, # Alterado para DEBUG para mais detalhes
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
    logger.debug(f"sanitize_text: Input text - {text[:100]}") # Log do início do texto
    text = text.replace('\r', ' ').replace('\n', ' ')
    sanitized = ' '.join(text.strip().split())
    logger.debug(f"sanitize_text: Sanitized text - {sanitized[:100]}") # Log do resultado
    return sanitized


def extract_text_from_pdf(file_url: str) -> list:
    """Extrai o texto de todas as páginas de um PDF."""
    logger.info(f"extract_text_from_pdf: Iniciando extração de {file_url}")
    all_text = []
    try:
        response = requests.get(file_url)
        response.raise_for_status() # Garante que a requisição foi bem-sucedida
        pdf_file = BytesIO(response.content)
        with pdfplumber.open(pdf_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if tables:
                    table_text = _process_tables(tables)
                    all_text.append((page_number, table_text))
                    logger.debug(f"extract_text_from_pdf: Tabela extraída da página {page_number}")
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    sanitized_text = sanitize_text(text)
                    all_text.append((page_number, sanitized_text))
                    logger.debug(f"extract_text_from_pdf: Texto da página {page_number} extraído.")
                else:
                    logger.warning(f"extract_text_from_pdf: Página {page_number} sem texto extraível.")
        logger.info(f"extract_text_from_pdf: Extração concluída. Total de páginas: {len(all_text)}")
        return all_text
    except requests.exceptions.RequestException as e:
        logger.error(f"extract_text_from_pdf: Erro ao baixar o PDF: {e}")
        raise Exception(f"Erro ao baixar o PDF: {e}")
    except pdfplumber.PDFOpeningError as e:
        logger.error(f"extract_text_from_pdf: Erro ao abrir o PDF: {e}")
        raise Exception(f"Erro ao abrir o PDF: {e}")


def _process_tables(tables: List[List[List[str]]]) -> str:
    """Processa tabelas extraídas do PDF."""

    table_strings = []
    for table in tables:
       # Simplificação: concatenar células com um separador
        table_string = "\n".join([" | ".join(row) for row in table])
        table_strings.append(f"Tabela:\n{table_string}")
    return "\n\n".join(table_strings)


def extract_references(text: str) -> list:
    """Extrai referências do texto (Art., §, Inciso, alínea)."""
    logger.debug(f"extract_references: Iniciando extração de referências em: {text[:100]}") # Log do início do texto
    references = []

   # Regex para Artigo (Art. 1º, Art. 1, Artigo 1)
    match = re.search(r'\b(Art(?:igo)?\.?\s*\d+[º°]?)', text, re.IGNORECASE)
    if match:
        references.append(match.group(1).strip())
        logger.debug(f"extract_references: Artigo encontrado: {match.group(1)}")

   # Regex para Parágrafo (§ 1º, Parágrafo único)
    match = re.search(r'\b(§\s*\d+[º°]?|Parágrafo(?:\s+único|\s+primeiro|segundo|terceiro)?)', text, re.IGNORECASE)
    if match:
        references.append(match.group(1).strip())
        logger.debug(f"extract_references: Parágrafo encontrado: {match.group(1)}")

   # Regex para Inciso (I, II, III...)
    match = re.search(r'\b([IVXLCDM]+)[).]', text)
    if match:
        references.append(f"Inciso {match.group(1)}")
        logger.debug(f"extract_references: Inciso encontrado: {match.group(1)}")

   # Regex para Alínea (a, b, c...)
    match = re.search(r'\b([a-z])[).]', text)
    if match:
        references.append(f"Alínea {match.group(1)}")
        logger.debug(f"extract_references: Alínea encontrada: {match.group(1)}")

    unique_references = list(dict.fromkeys(references)) # Remove duplicatas mantendo a ordem
    logger.debug(f"extract_references: Referências extraídas: {unique_references}")
    return unique_references


def split_text_into_chunks(text: str, page_number: int, parent_metadata: Dict = None) -> list:
    """Divide recursivamente o texto em chunks menores, identificando seções por marcadores."""

    logger.debug(f"split_text_into_chunks: Iniciando divisão de texto em chunks: {text[:100]}") # Log do início do texto

    chunks = []
    if parent_metadata is None:
        parent_metadata = {}

   # 1. Chunking Semântico (Títulos)
    title_chunks = _chunk_by_titles_recursive(text, page_number, parent_metadata)
    if title_chunks:
        return title_chunks # Retorna se títulos forem encontrados

   # 2. Chunking por Parágrafos
    para_chunks = _chunk_by_paragraphs(text, page_number, parent_metadata)
    if para_chunks:
        return para_chunks

   # 3. Chunking por Frases
    sentence_chunks = _chunk_by_sentences(text, page_number, parent_metadata)
    return sentence_chunks


def _chunk_by_titles_recursive(text: str, page_number: int, parent_metadata: Dict) -> List[Dict]:
    """Divide o texto recursivamente usando títulos e cabeçalhos como delimitadores."""

    title_pattern = re.compile(
        r'(\b(?:CAPÍTULO|SEÇÃO|Art\.\s*\d+|[A-Z][a-z]+\s+\d+)\b.*?)(?=\b(?:CAPÍTULO|SEÇÃO|Art\.\s*\d+|[A-Z][a-z]+\s+\d+)\b|$)',
        re.IGNORECASE | re.DOTALL
    )
    matches = list(title_pattern.finditer(text))
    if not matches:
        return []

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        title = matches[i].group(1).strip()
        current_metadata = {"type": "title", "title": title}
        current_metadata.update(parent_metadata) # Herda metadados

        if len(chunk_text) > 300:
           # Recursão!
            sub_chunks = split_text_into_chunks(chunk_text, page_number, current_metadata)
            chunks.extend(sub_chunks)
        elif len(chunk_text) > 50:
            chunks.append({
                "page": page_number,
                "text": chunk_text,
                "metadata": current_metadata
            })
    return chunks


def _chunk_by_paragraphs(text: str, page_number: int, parent_metadata: Dict) -> List[Dict]:
    """Divide o texto usando parágrafos como delimitadores."""

    para_pattern = re.compile(r'(.+?\n\n)', re.DOTALL)
    matches = list(para_pattern.finditer(text))
    if not matches:
        return []

    chunks = []
    for match in matches:
        chunk_text = match.group(1).strip()
        if 50 < len(chunk_text) < 500: # Limites de tamanho
            current_metadata = {"type": "paragraph"}
            current_metadata.update(parent_metadata)
            chunks.append({
                "page": page_number,
                "text": chunk_text,
                "metadata": current_metadata
            })
    return chunks


def _chunk_by_sentences(text: str, page_number: int, parent_metadata: Dict) -> List[Dict]:
    """Divide o texto em frases, garantindo que os chunks não excedam um tamanho máximo."""

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 300: # Limite de tamanho
            current_chunk += sentence + " "
        else:
            if current_chunk:
                current_metadata = {"type": "sentence"}
                current_metadata.update(parent_metadata)
                chunks.append({
                    "page": page_number,
                    "text": current_chunk.strip(),
                    "metadata": current_metadata
                })
            current_chunk = sentence + " "
    if current_chunk:
        current_metadata = {"type": "sentence"}
        current_metadata.update(parent_metadata)
        chunks.append({
            "page": page_number,
            "text": current_chunk.strip(),
            "metadata": current_metadata
        })
    return chunks


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """Gera o embedding para o texto usando o modelo da OpenAI."""
    logger.debug(f"get_embedding: Obtendo embedding para: {text[:100]}") # Log do início do texto

    if not text.strip():
        logger.error("get_embedding: Texto do chunk está vazio!")
        raise ValueError("Texto do chunk está vazio!")
    response = openai.embeddings.create(model=model, input=text)
    embedding = response.data[0].embedding
    logger.debug(f"get_embedding: Embedding obtido: {embedding[:10]}") # Log dos primeiros valores
    return embedding


def generate_chunk_hash(chunk_text: str) -> str:
    """Gera um hash para o texto do chunk."""
    logger.debug(f"generate_chunk_hash: Gerando hash para: {chunk_text[:100]}") # Log do início do texto

    chunk_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
    logger.debug(f"generate_chunk_hash: Hash gerado: {chunk_hash}")
    return chunk_hash


def check_chunk_exists(chunk_hash: str) -> bool:
    """Verifica se um chunk com o hash especificado já existe no banco de dados."""
    logger.debug(f"check_chunk_exists: Verificando existência do chunk com hash: {chunk_hash}")

    response = supabase.table("pdf_embeddings_textos").select("id").eq("chunk_hash", chunk_hash).execute()
    exists = bool(response.data)
    logger.debug(f"check_chunk_exists: Chunk existe: {exists}")
    return exists


def insert_embeddings_to_supabase(chunks_with_metadata: list) -> None:
    """Insere os embeddings e metadados dos chunks no Supabase."""
    logger.info(f"insert_embeddings_to_supabase: Inserindo {len(chunks_with_metadata)} chunks no Supabase")

    for i, item in enumerate(chunks_with_metadata):
        if check_chunk_exists(item["chunk_hash"]):
            logger.warning(f"insert_embeddings_to_supabase: Chunk {i + 1} já existe. Pulando.")
            continue

        try:
            response = supabase.table("pdf_embeddings_textos").insert(item).execute()
            if response.data:
                logger.info(f"insert_embeddings_to_supabase: Chunk {i + 1} inserido com sucesso!")
            else:
                logger.error(f"insert_embeddings_to_supabase: Erro ao inserir chunk {i + 1}. Detalhes: {response.error}")
        except Exception as e:
            logger.error(f"insert_embeddings_to_supabase: Erro ao inserir chunk {i + 1}. Detalhes: {e}")


def vectorize_pdf(file_url: str, condominio_id: str) -> list:
    """Processa o PDF, extrai o texto, divide em chunks, gera embeddings e salva no Supabase."""
    logger.info(f"vectorize_pdf: Iniciando vetorização do PDF: {file_url}")

    nome_documento = os.path.basename(file_url)
    origem = "upload_local"
    logger.info("vectorize_pdf: Extraindo texto do PDF...")
    pages = extract_text_from_pdf(file_url)

    if not pages:
        logger.warning("vectorize_pdf: Nenhum texto extraído.")
        return []

    all_chunks = []
    for page_number, page_text in pages:
        logger.info(f"vectorize_pdf: Página {page_number}: extraindo por marcadores...")
        chunks = split_text_into_chunks(page_text, page_number) # Chamada inicial para chunking recursivo
        logger.info(f"vectorize_pdf: Chunks detectados: {len(chunks)}")

        for chunk in chunks:
            chunk_text = chunk["text"]
            references = chunk.get("references", []) # Usar get() para evitar erro se "references" não existir
            chunk_hash = generate_chunk_hash(chunk_text)

            try:
                embedding = get_embedding(chunk_text)
            except Exception as e:
                logger.error(f"vectorize_pdf: Erro ao gerar embedding: {e}")
                embedding = None

            all_chunks.append({
                "condominio_id": condominio_id,
                "nome_documento": nome_documento,
                "origem": origem,
                "pagina": chunk.get("page", page_number), # Usar get() para segurança
                "chunk_text": chunk_text,
                "chunk_hash": chunk_hash,
                "embedding": embedding,
                "referencia_detectada": " | ".join(references)
            })
        time.sleep(0.5) # Adiciona um pequeno delay entre o processamento de cada página
    logger.info(f"vectorize_pdf: Vetorização concluída. Total de chunks: {len(all_chunks)}")
    return all_chunks


# Rotas da API
@app.get("/")
def home():
    return {"message": "FastAPI está funcionando!"}


@app.post("/vetorizar")
async def vetorizar_pdf(item: Item):
    """Endpoint para iniciar a vetorização de um PDF."""
    logger.info(f"vetorizar_pdf: Iniciando vetorização para: {item.file_url}, {item.condominio_id}")

    file_url = item.file_url
    condominio_id = item.condominio_id

    if not file_url or not condominio_id:
        logger.error("vetorizar_pdf: Parâmetros obrigatórios ausentes")
        raise HTTPException(status_code=400, detail="Parâmetros 'file_url' e 'condominio_id' são obrigatórios.")

    nome_documento = os.path.basename(file_url)

    try:
       # Verifica se já existe um registro para este condomínio e documento
        verifica = supabase.table("pdf_artigos_extraidos").select("id").eq("condominio_id", condominio_id).eq(
            "nome_documento", nome_documento).execute()
        if not verifica.data:
            logger.info("vetorizar_pdf: Inserindo novo registro em pdf_artigos_extraidos")
            supabase.table("pdf_artigos_extraidos").insert({
                "condominio_id": condominio_id,
                "nome_documento": nome_documento,
                "status": "pendente"
            }).execute()

       # Deleta os registros existentes para este condomínio e documento
        supabase.table("pdf_embeddings_textos").delete().eq("condominio_id", condominio_id).eq(
            "nome_documento", nome_documento).execute()

        logger.info(f"vetorizar_pdf: Iniciando processamento do PDF: {file_url}")
        vectorized_data = vectorize_pdf(file_url, condominio_id)

        if vectorized_data:
            logger.info(f"vetorizar_pdf: Inserindo {len(vectorized_data)} chunks no Supabase")
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
            logger.warning("vetorizar_pdf: Nenhum dado foi vetorizado")
            raise HTTPException(status_code=400, detail="Nenhum dado foi extraído do PDF.")

    except Exception as e:
        logger.exception(f"vetorizar_pdf: Erro não esperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


if __name__ == "__main__":
    print("Verificando conexões com serviços externos...")
   # Você pode adicionar aqui verificações para Supabase e OpenAI
   # e usar sys.exit(0) em caso de falha
