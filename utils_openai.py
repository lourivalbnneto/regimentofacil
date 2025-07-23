import os
import openai
import logging

logger = logging.getLogger("utils_openai")

openai.api_key = os.getenv("OPENAI_API_KEY")

async def gerar_embeddings_para_chunks(chunks):
    for chunk in chunks:
        try:
            resposta = await openai.embeddings.async_create(
                model="text-embedding-3-small",
                input=chunk["chunk_text"]
            )
            chunk["embedding"] = resposta.data[0].embedding
            logger.info("✅ Embedding gerado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {e}")
            chunk["embedding"] = None
    return chunks
