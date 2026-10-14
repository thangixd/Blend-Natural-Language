FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --all-extras
RUN .venv/bin/python -c "import duckdb; duckdb.connect().install_extension('vss')"

COPY . .

RUN .venv/bin/python -c "import duckdb; duckdb.connect('blend_duckdb.db').close()"

ENV PATH="/app/.venv/bin:${PATH}"
CMD ["tail", "-f", "/dev/null"]
