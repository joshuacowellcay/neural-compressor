FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install CPU-only torch first so the bigger pieces are cached separately from app code.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install -r requirements.txt \
    && pip install uvicorn[standard]==0.32.1

# App code
COPY pyproject.toml ./
COPY src ./src
COPY backend ./backend
COPY models ./models
COPY corpus ./corpus
COPY scripts ./scripts

RUN pip install --no-deps .

EXPOSE 8000

CMD ["uvicorn", "backend.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
