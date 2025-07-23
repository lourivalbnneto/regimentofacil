import os
import json
from typing import List, Dict
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

def supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

async def salvar_chunks_no_supabase(chunks: List[Dict]):
    supabase = supabase_client()
    inseridos = 0
    erros = 0

    for chunk in chunks:
        try:
            print(f"[Inserção] Tentando inserir chunk:\n{json.dumps(chunk, ensure_ascii=False)[:500]}...\n---")
            data = supabase.table("pdf_embeddings_textos").insert(chunk).execute()
            print(f"[Sucesso] Inserido com sucesso: {data}\n")
            inseridos += 1
        except Exception as e:
            print(f"[Erro] Falha ao inserir chunk:\n{e}\nChunk problemático:\n{json.dumps(chunk, ensure_ascii=False)}\n")
            erros += 1

    print(f"[Resumo] Total de chunks inseridos com sucesso: {inseridos}")
    print(f"[Resumo] Total de erros de inserção: {erros}")
