FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-baixar recursos do NLTK durante a construção da imagem
RUN python -c "import nltk; nltk.download('punkt')"

COPY . .

EXPOSE 5000

CMD ["sh", "-c", "uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}"]