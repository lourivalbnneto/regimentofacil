FROM python:3.11-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /web

# Copia arquivos
COPY . .

# Instala dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposição da porta
EXPOSE 5000

# Comando para iniciar o servidor
ENTRYPOINT ["uvicorn", "vectorize_pdf:app", "--host=0.0.0.0", "--port=5000"]
