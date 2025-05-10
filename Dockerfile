FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["sh", "-c", "uvicorn app:app --host=0.0.0.0 --port=$PORT"]
