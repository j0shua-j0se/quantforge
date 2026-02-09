FROM python:3.10.19-slim

WORKDIR /quantforge

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

ENV PYTHONUNBUFFERED=1
ENV SEED=42

CMD ["python"]
