import os
import openai
import json
import time
import hashlib
import pdfplumber
import nltk
import re
from supabase import create_client, Client
from dotenv import load_dotenv
from io import BytesIO
import requests
from flask import Flask, request, jsonify

# 🔹 Carregar variáveis de ambiente do .env
load_dotenv()

# 🔹 Configuração das chaves
openai.api_key = os.getenv("OPENAI_API_KEY")  # Chave da OpenAI
SUPABASE_URL = os.getenv("SUPABASE_URL")  # URL do Supabase
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Chave de serviço do Supabase

# 🔹 Criando o cliente do Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🔹 Garantir que o tokenizer do NLTK esteja disponível
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# 🔹 Função utilitária para limpar quebras de linha e múltiplos espaços
def limpar_texto(texto):
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(texto.split())

# 🔹 Função para extrair texto do PDF por página
def extract_text_from_pdf(file_url):
    all_text = []
    response = requests.get(file_url)  # Baixa o PDF da URL fornecida
    if response.status_code != 200:
        raise Exception(f"Erro ao baixar o PDF: {response.status_code}")
    
    pdf_file = BytesIO(response.content)  # Converte o conteúdo para um objeto BytesIO
    with pdfplumber.open(pdf_file) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                all_text.append((page_number, text.strip()))
            else:
                print(f"⚠️ Página {page_number} sem texto extraível.")
    return all_text

# 🔹 Dividir texto por artigos (usando regex para detectar "Art.")
def split_by_articles(text):
    pattern = r'(Art(?:igo)?\.?\s*\d+[ºo]?)'
    split_parts = re.split(pattern, text)

    articles = []
    for i in range(1, len(split_parts), 2):
        artigo_numero = split_parts[i].strip()
        artigo_texto = split_parts[i + 1].strip() if i + 1 < len(split_parts) else ''
        articles.append(f"{artigo_numero} {artigo_texto}")
    return articles

# 🔹 Gerar embedding com OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    if not text.strip():
        raise ValueError("Texto do chunk está vazio!")
    response = openai.embeddings.create(model=model, input=text)
    return response.data[0].embedding

# 🔹 Gerar hash do chunk
def generate_chunk_hash(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()

# 🔹 Verificar se o chunk já existe na tabela do Supabase
def check_chunk_exists(chunk_hash):
    response = supabase.table("pdf_embeddings_textos").select("id").eq("chunk_hash", chunk_hash).execute()
    return bool(response.data)

# 🔹 Inserir chunks no Supabase
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

# 🔹 Função para processar o PDF e gerar os embeddings
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
            chunk = limpar_texto(article.strip())  # Limpeza de texto
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
            time.sleep(0.5)  # Evitar rate limit da OpenAI
    return all_chunks

# 🔹 Inicializando o Flask para criar a API
app = Flask(__name__)

@app.route('/vetorizar', methods=['POST'])
def vetorizar_pdf_endpoint():
    try:
        # Receber o JSON da requisição
        data = request.get_json()
        file_url = data.get("file_url")
        condominio_id = data.get("condominio_id")

        # Validar dados
        if not file_url or not condominio_id:
            return jsonify({"error": "Parâmetros 'file_url' e 'condominio_id' são obrigatórios."}), 400

        # Processar o PDF e gerar os embeddings
        vectorized_data = vectorize_pdf(file_url, condominio_id)

        # Salvar os embeddings no Supabase
        if vectorized_data:
            insert_embeddings_to_supabase(vectorized_data)
            return jsonify({"message": "Vetorização e inserção completadas com sucesso!"}), 200
        else:
            return jsonify({"error": "Falha na vetorização."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)