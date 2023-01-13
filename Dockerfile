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
# COPY .git/ .git/
COPY data.dvc data.dvc

WORKDIR /
RUN pip install -r requirements_cuda.txt --no-cache-dir
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://butterfly_jar/
RUN dvc pull 
RUN wanbd login(key=5df58a0e3f5189c3a99e6c0a1afc0f107a3519d9)
RUN python -c "import wandb; wandb.login(key='5df58a0e3f5189c3a99e6c0a1afc0f107a3519d9')"

ENTRYPOINT ["python", "-u", "src/models/train_model_PL.py"]
