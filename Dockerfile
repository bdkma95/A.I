# Build stage
FROM python:3.11-slim-bookworm as builder

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with vulnerability scanning
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt \
    && pip check --disable-pip-version-check


# Runtime stage
FROM python:3.11-slim-bookworm

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1

# Create secure user
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

# Security hardening
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tini \
    libcap2-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && setcap 'cap_net_bind_service=+ep' /usr/local/bin/python3.11

# Copy from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . .

# Security configuration
RUN find /app -type d -exec chmod 755 {} \; \
    && find /app -type f -exec chmod 644 {} \; \
    && chmod 755 /usr/local/bin/python3.11

USER appuser

# Security ports
EXPOSE 5000

# Runtime protections
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "dashboard:app"]
