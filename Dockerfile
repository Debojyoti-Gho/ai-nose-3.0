# Dockerfile - deterministic build, avoids Nixpacks timeout
FROM python:3.11-slim

WORKDIR /app

# system deps for opencv/shap etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libgl1 && \
    rm -rf /var/lib/apt/lists/*

# copy only requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
# install lighter deps first (fast)
RUN pip install -r requirements.txt

# copy rest of repo
COPY . /app

# Ensure port env var available to container
ENV PORT=8000
EXPOSE 8000

# start command (Railway can override)
CMD ["uvicorn", "ai_nose_api:app", "--host", "0.0.0.0", "--port", "8000"]
