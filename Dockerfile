# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_cuda.txt requirements_cuda.txt
COPY setup.py setup.py
COPY src/ src/
# COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY conf/ conf/

WORKDIR /
RUN pip install -r requirements_cuda.txt --no-cache-dir
RUN dvc pull

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
