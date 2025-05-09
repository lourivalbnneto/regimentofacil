from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()  # Carregar variáveis de ambiente

# Criando a aplicação FastAPI
app = FastAPI()

# Definindo o modelo de dados para a requisição
class Item(BaseModel):
    file_url: str
    condominio_id: str

@app.get("/")
def home():
    return {"message": "FastAPI está funcionando!"}

@app.post("/vetorizar")
async def vetorizar_pdf(item: Item):
    try:
        file_url = item.file_url
        condominio_id = item.condominio_id

        if not file_url or not condominio_id:
            return {"error": "Parâmetros 'file_url' e 'condominio_id' são obrigatórios."}, 400

        # Aqui vai a lógica da vetorização

        return {"message": "Vetorização completada com sucesso!"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
