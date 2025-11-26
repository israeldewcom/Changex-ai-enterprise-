FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r production.txt

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 changex && chown -R changex:changex /app
USER changex

# Create upload directory
RUN mkdir -p /app/uploads && chmod 755 /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
