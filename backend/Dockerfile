FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
# 專案 label:make clean 只清帶它的 image;dev / runtime 從 base 繼承
LABEL org.opencontainers.image.title="mdrsmap-classifier"
WORKDIR /app
COPY pyproject.toml uv.lock ./
ENV PATH="/app/.venv/bin:$PATH"

# dev:開發與測試用,含 pytest 與 ruff,不出貨
FROM base AS dev
RUN uv sync --frozen
COPY app ./app
COPY mdrsmap ./mdrsmap
COPY tests ./tests
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# runtime:出貨用。必須留在最後:沒指定 target 時 Docker 建最後一個 stage,
# 調換順序會讓含測試工具的 dev image 被當成出貨的 image
FROM base AS runtime
RUN uv sync --frozen --no-dev
COPY app ./app
COPY mdrsmap ./mdrsmap
RUN useradd --create-home --uid 1000 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
