FROM python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libeigen3-dev \
    zlib1g-dev \
    libboost-dev \
    libopenbabel-dev \
    swig \
    libxml2-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install fpocket from GitHub
RUN echo "Installing fpocket..." && \
    wget https://github.com/Discngine/fpocket/archive/refs/tags/3.1.5.tar.gz -O fpocket-3.1.5.tar.gz && \
    tar -xzf fpocket-3.1.5.tar.gz && \
    cd fpocket-3.1.5 && \
    make && \
    make install PREFIX=/usr && \
    cd .. && \
    rm -rf fpocket-3.1.5 fpocket-3.1.5.tar.gz && \
    echo "fpocket installation completed" || \
    { echo "ERROR: fpocket installation failed"; exit 1; }

# Install Open Babel
RUN echo "Installing Open Babel..." && \
    wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1.tar.gz -O openbabel-3.1.1.tar.gz && \
    tar -xzf openbabel-3.1.1.tar.gz && \
    cd openbabel-3.1.1 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make && \
    make install && \
    cd ../.. && \
    rm -rf openbabel-3.1.1 openbabel-3.1.1.tar.gz && \
    echo "Open Babel installation completed" || \
    { echo "ERROR: Open Babel installation failed"; exit 1; }

# Verify installations and permissions
RUN ls -l /usr/bin/fpocket && \
    /usr/bin/fpocket -h && \
    ls -l /usr/bin/obabel && \
    /usr/bin/obabel -V || \
    { echo "ERROR: fpocket or obabel not found or not executable"; exit 1; }

# Ensure binaries are executable by all users
RUN chmod +x /usr/bin/fpocket /usr/bin/obabel

# Set PATH explicitly
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"

# Create static directory with correct permissions
RUN mkdir -p /opt/render/project/src/static && \
    chown -R 1000:1000 /opt/render/project/src/static && \
    chmod -R 755 /opt/render/project/src/static

# Set non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--timeout", "600", "--workers", "1", "--threads", "4", "app:app"]