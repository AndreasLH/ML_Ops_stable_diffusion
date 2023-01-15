FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY api.py api.py
COPY src/ src/

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn


CMD exec uvicorn api:app --port $PORT --host 0.0.0.0 --workers 1
