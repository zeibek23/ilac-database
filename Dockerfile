# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    unzip \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# Install fpocket2
RUN wget https://sourceforge.net/projects/fpocket/files/fpocket2.tar.gz/download -O fpocket2.tar.gz && \
    tar -xzf fpocket2.tar.gz && \
    cd fpocket2 && \
    make && \
    make install && \
    cd .. && \
    rm -rf fpocket2 fpocket2.tar.gz

# Verify installations
RUN which fpocket && fpocket -h && \
    which obabel && obabel -V || { echo "ERROR: fpocket or obabel not found"; exit 1; }

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
CMD ["/bin/bash", "-c", "echo 'Runtime PATH: $PATH' && which fpocket && which obabel && exec gunicorn -b 0.0.0.0:$PORT --timeout 1200 --workers 4 --threads 4 app:app"]