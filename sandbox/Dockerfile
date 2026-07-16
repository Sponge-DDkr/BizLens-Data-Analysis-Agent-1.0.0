# BizLens Sandbox Image
# Python 3.12 + Pandas + DuckDB + OpenPyXL
# Run: docker build -t bizlens-sandbox .
# Verify: docker run --rm bizlens-sandbox python -c "import pandas; print('ok')"

FROM python:3.12-slim

LABEL description="BizLens Data Analysis Sandbox — pre-installed pandas/duckdb/openpyxl"

# Security: create non-root user
RUN useradd --create-home --shell /bin/bash sandbox

# Install only needed packages — no git, curl, or network tools
RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    duckdb==1.1.3 \
    openpyxl==3.1.5 \
    plotly==5.24.1 \
    numpy==2.1.3

# Switch to unprivileged user
USER sandbox
WORKDIR /data

# Entrypoint: run Python code passed as argument
ENTRYPOINT ["python", "-c"]
