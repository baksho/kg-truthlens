FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[anthropic]"

ENV GRAPHRAG_KG_BACKEND=http
EXPOSE 8080
CMD ["uvicorn", "graphrag_pharmaco.api:app", "--host", "0.0.0.0", "--port", "8080"]
