# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# Verify OpenBabel installation
RUN which obabel && obabel -V && obabel -L filters | grep autosite || { echo "ERROR: obabel or autosite not found"; exit 1; }

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==22.0.0

# Pre-download NLTK data
RUN python -c 'import nltk; nltk.download("brown"); nltk.download("punkt")'

# Copy application code
COPY . .

# Create static directory with correct permissions
RUN mkdir -p /opt/render/project/src/static && \
    chmod -R 755 /opt/render/project/src/static

# Create app user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /opt/render/project/src/static

# Switch to appuser
USER appuser

# Ensure /app is writable
RUN chmod -R u+rw /app

# Expose port
EXPOSE 8000

# Debug PATH and binaries at runtime
CMD ["/bin/bash", "-c", "echo 'Runtime PATH: $PATH' && which obabel && obabel -V && exec gunicorn -b 0.0.0.0:$PORT --timeout 1200 --workers 4 --threads 4 app:app"]