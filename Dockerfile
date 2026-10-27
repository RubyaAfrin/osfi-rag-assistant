FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OSFI_INDEX_DIR=/app/index \
    HF_HOME=/app/.cache/huggingface

# CPU-only PyTorch keeps the image much smaller than the default GPU build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY osfi_rag ./osfi_rag
COPY scripts ./scripts

# The search index is mounted at runtime (see docker-compose.yml), not baked into the image
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "osfi_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
