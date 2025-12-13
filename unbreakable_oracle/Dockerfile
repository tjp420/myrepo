# Dockerfile for Unbreakable Oracle RAG prototype
# Build with: docker build -f unbreakable_oracle/Dockerfile -t unbreakable_oracle:local .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only requirements first for caching
COPY unbreakable_oracle/requirements.txt /app/unbreakable_oracle/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/unbreakable_oracle/requirements.txt

# copy repository
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "unbreakable_oracle.api:app", "--host", "0.0.0.0", "--port", "8000"]
