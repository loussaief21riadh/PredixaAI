FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system predixa \
    && adduser --system --ingroup predixa predixa

COPY requirements.txt requirements-staging.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --requirement requirements-staging.txt

COPY . .

RUN mkdir -p \
        /app/logs \
        /app/reports \
        /app/trained_models \
    && chown -R predixa:predixa /app

USER predixa

EXPOSE 8000

CMD ["sh", "scripts/start_staging.sh"]
