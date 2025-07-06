# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Open Babel build
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    unzip \
    cmake \
    libxml2-dev \
    zlib1g-dev \
    libeigen3-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Open Babel from source
RUN wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1.tar.gz && \
    tar -xzf openbabel-3.1.1.tar.gz && \
    cd openbabel-3.1.1 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && make install && \
    cd ../../ && rm -rf openbabel-3.1.1 openbabel-3.1.1.tar.gz && \
    ldconfig

# Install fpocket2
RUN wget https://sourceforge.net/projects/fpocket/files/fpocket2.tar.gz/download -O fpocket2.tar.gz && \
    tar -xzf fpocket2.tar.gz && \
    mv fpocket* fpocket2 && \
    cd fpocket2 && \
    make && \
    make install && \
    cd .. && \
    rm -rf fpocket2 fpocket2.tar.gz

# Verify Open Babel and fpocket during build
RUN which obabel && obabel -V || { echo "ERROR: obabel not found or not executable"; exit 1; } && \
    ls -l /usr/local/bin/obabel || { echo "ERROR: obabel binary missing in /usr/local/bin"; exit 1; } && \
    which fpocket && fpocket -h || { echo "ERROR: fpocket not found"; exit 1; }

# Ensure obabel is executable and accessible
RUN chmod +x /usr/local/bin/obabel && \
    ln -sf /usr/local/bin/obabel /usr/bin/obabel

# Install Python dependencies
COPY requirements.txt .
RUN rm -rf /tmp/pip*  # Clear stale pip artifacts
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir numpy==1.23.5  # Pre-install numpy
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt
RUN pip install --no-cache-dir gunicorn==22.0.0

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

# Debug PATH, obabel, and fpocket at runtime
CMD ["/bin/bash", "-c", "echo 'Runtime PATH: $PATH'; ls -l /usr/local/bin/obabel 2>/dev/null || echo 'obabel not found in /usr/local/bin'; ls -l /usr/bin/obabel 2>/dev/null || echo 'obabel not found in /usr/bin'; which obabel; obabel -V || echo 'obabel failed to run'; which fpocket; exec gunicorn -b 0.0.0.0:$PORT --timeout 1200 --workers 4 --threads 4 app:app"]