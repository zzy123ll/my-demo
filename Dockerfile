# Enterprise RAG CS - Docker Image (HuggingFace 模型预缓存)
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    langchain-core langchain-openai \
    chromadb sentence-transformers \
    numpy==1.26.4 \
    pyyaml \
    pytest pytest-asyncio \
    httpx \
    spacy \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# === Pre-download HuggingFace models into image layer ===
# Layer 1: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); print('paraphrase-multilingual-MiniLM-L12-v2: OK')"

# Layer 2: bge-reranker-v2-m3
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512); print('bge-reranker-v2-m3: OK')"

# Layer 3: mDeBERTa-v3-base-mnli-xnli (NLI)
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoTokenizer.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'); AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'); print('mDeBERTa-v3-base-mnli-xnli: OK')"

# Layer 4: spaCy zh_core_web_sm
RUN python -m spacy download zh_core_web_sm && python -c "import spacy; spacy.load('zh_core_web_sm'); print('zh_core_web_sm: OK')"

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p logs chroma_data data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
