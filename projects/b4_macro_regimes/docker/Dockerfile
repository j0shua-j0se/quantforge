FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir pandas==2.2.3 numpy==1.26.4 pyyaml==6.0.2 hmmlearn==0.3.0 scikit-learn==1.3.2 fredapi==0.5.1

COPY core/ /app/core/
COPY projects/b4_macro_regimes/ /app/projects/b4_macro_regimes/

RUN mkdir -p /app/data/macro /app/data/regimes

CMD ["python", "projects/b4_macro_regimes/pipeline.py"]

LABEL maintainer="joshua.jose2002@gmail.com"
LABEL description="B4 Macro Regime Detection"
LABEL version="1.0.0"
