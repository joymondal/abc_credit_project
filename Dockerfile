# ABC Credit loan chatbot - container image.
#   docker build -t abc-credit-bot:1.0 .
#   docker run --rm -p 8501:8501 --env-file .env -v abc_audit:/app/data abc-credit-bot:1.0
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ABC_AUDIT_DIR=/app/data
# libgomp1 = OpenMP runtime LightGBM needs on slim images; curl is used by the health check
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY chatbot/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY --chown=appuser:appuser abc_features.py ./
COPY --chown=appuser:appuser chatbot ./chatbot
COPY --chown=appuser:appuser artifacts ./artifacts
COPY --chown=appuser:appuser .streamlit ./.streamlit
# the audit repository lives on a mounted volume so that it survives restarts
RUN mkdir -p /app/data && chown appuser:appuser /app/data
USER appuser
EXPOSE 8501
# Hosts such as Render and Cloud Run tell the container which port to listen on through $PORT; locally it is 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1
CMD ["sh", "-c", "streamlit run chatbot/app_streamlit.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
