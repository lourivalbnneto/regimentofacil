# Dockerfile

FROM python:3.11-slim

# Instala dependências do sistema para pdfplumber e NLTK
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório da aplicação
WORKDIR /app

# Copia arquivos
COPY . .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Porta para o Uvicorn
EXPOSE 5000

# Comando para iniciar o app
CMD ["uvicorn", "vectorize_pdf_railway:app", "--host", "0.0.0.0", "--port", "5000"]
