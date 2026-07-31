# Small CPU-only image for the inference API (HF Spaces / Render free tier).
# The gate is a tiny MLP — no GPU needed — so we pull CPU-only torch wheels
# to keep the image well under free-tier size limits.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first (the default PyPI wheel bundles CUDA and is
# several GB); everything else then installs from PyPI as usual and is a
# no-op for torch since the >=2.1 constraint is already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Checkpoints are not baked into the image; mount/download them and point
# MODEL_DIR at that folder. Until then /predict returns a 503, everything
# else works.
ENV MODEL_DIR=/app/data/models
ENV ALLOWED_ORIGINS=*

# Render/HF Spaces inject PORT; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
