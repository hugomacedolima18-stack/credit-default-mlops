# One image is shared by the three services (mlflow, trainer, api).
# Each service only changes the command it runs (see docker-compose.yml).
FROM python:3.11-slim

# Print logs immediately and do not write .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

# Install dependencies first (this layer is cached between builds).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code.
COPY src/ src/
COPY api/ api/
COPY tests/ tests/
COPY pytest.ini .

# Folders that Docker Compose mounts at runtime.
RUN mkdir -p data/raw artifacts

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
