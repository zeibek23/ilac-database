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
    && rm -rf /var/lib/apt/lists/*
# Install fpocket
RUN wget https://downloads.sourceforge.net/project/fpocket/fpocket-3.1.5.tar.gz && \
    tar -xzf fpocket-3.1.5.tar.gz && \
    cd fpocket-3.1.5 && \
    make && \
    make install PREFIX=/usr && \  # Install to /usr/bin
    cd .. && rm -rf fpocket-3.1.5 fpocket-3.1.5.tar.gz
# Install Open Babel
RUN wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1.tar.gz && \
    tar -xzf openbabel-3.1.1.tar.gz && \
    cd openbabel-3.1.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \  # Install to /usr/bin
    make && make install && \
    cd ../.. && rm -rf openbabel-3.1.1 openbabel-3.1.1.tar.gz
# Verify installations
RUN which fpocket || { echo "ERROR: fpocket not found"; exit 1; }
RUN which obabel || { echo "ERROR: obabel not found"; exit 1; }
RUN obabel -V && fpocket -h  # Additional verification
# Set PATH explicitly
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"
# Create static directory with write permissions
RUN mkdir -p /opt/render/project/src/static && chmod -R 777 /opt/render/project/src/static
# Install Python dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY . /app
EXPOSE 8080
CMD ["gunicorn", "--timeout", "600", "--workers", "1", "--threads", "4", "app:app"]