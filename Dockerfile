# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Runtime stage
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH=/home/user/.local/bin:$PATH

# Create non-root user
RUN useradd --create-home user && \
    chown -R user:user /app
USER user

# Copy from builder
COPY --from=builder --chown=user:user /root/.local /home/user/.local
COPY --chown=user:user . .

# Application ports
EXPOSE 5000  # Dashboard
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:5000 || exit 1

# Runtime command (override with docker-compose)
CMD ["python", "-m", "main"]
