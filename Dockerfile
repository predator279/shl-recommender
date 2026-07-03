# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for FAISS + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding model into the image at build time
# (avoids downloading at runtime inside the 30s call budget)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY app/ ./app/

# Copy prebuilt FAISS index + metadata sidecar
# These must be built BEFORE docker build (run: python scripts/build_index.py)
COPY data/catalog_index.faiss ./data/catalog_index.faiss
COPY data/catalog_metadata.json ./data/catalog_metadata.json

# Expose the port uvicorn listens on
EXPOSE 8000

# Run uvicorn — models load at startup (lifespan event), not lazily
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
