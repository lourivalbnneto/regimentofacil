from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from utils_pdf import extract_and_chunk_pdf
from utils_openai import gerar_embeddings_para_chunks
from utils_db import salvar_chunks_no_supabase  # <- Certifique-se de importar corretamente

app = FastAPI()

# CORS liberado para testes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

@app.get("/")
def read_root():
    return {"message": "API Vetorizador ativa"}

@app.post("/vetorizar")
async def vetorizar(request: Request):
    try:
        body = await request.json()
        url_pdf = body.get("url_pdf")
        nome_documento = body.get("nome_documento")
        condominio_id = body.get("condominio_id")
        id_usuario = body.get("id_usuario")
        origem = body.get("origem", "upload_manual")

        if not url_pdf or not condominio_id or not id_usuario:
            return {"success": False, "message": "Campos obrigatórios ausentes"}

        # Etapa 1: Extração + chunking
        chunks = await extract_and_chunk_pdf(
            url_pdf=url_pdf,
            nome_documento=nome_documento,
            condominio_id=condominio_id,
            id_usuario=id_usuario,
            origem=origem
        )

        print(f"🔍 Total de chunks extraídos: {len(chunks)}")

        if not chunks:
            return {"success": False, "message": "Nenhum chunk gerado"}

        # Etapa 2: Geração de embeddings
        chunks = await gerar_embeddings_para_chunks(chunks)

        print("🧠 Verificando conteúdo dos chunks:")
        for i, c in enumerate(chunks):
            print(f"{i+1}. Texto: {c.get('chunk_text')[:50]}... | Embedding: {'ok' if 'embedding' in c else '❌'}")

        # Etapa 3: Salvar no Supabase
        print(f"💾 Enviando {len(chunks)} chunks para o Supabase")

        inseridos = salvar_chunks_no_supabase(chunks)
        logging.info(f"✅ {inseridos} chunks inseridos no Supabase")

        return {"success": True, "chunks_salvos": inseridos}
    except Exception as e:
        logging.error(f"Erro ao processar PDF: {e}")
        return {"success": False, "message": str(e)}
