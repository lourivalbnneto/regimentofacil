FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["sh", "-c", "ls -la && pwd && python3 -c 'import vectorize_pdf; print(vectorize_pdf.app)' && uvicorn vectorize_pdf:app --host=0.0.0.0 --port=$PORT"]
