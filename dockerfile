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
RUN echo "Installing fpocket..." && \
    wget https://downloads.sourceforge.net/project/fpocket/fpocket-3.1.5.tar.gz && \
    tar -xzf fpocket-3.1.5.tar.gz && \
    cd fpocket-3.1.5 && \
    make && \
    make install PREFIX=/usr && \  # Install to /usr/bin
    cd .. && rm -rf fpocket-3.1.5 fpocket-3.1.5.tar.gz || { echo "ERROR: fpocket installation failed"; exit 1; }
# Install Open Babel
RUN echo "Installing Open Babel..." && \
    wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1.tar.gz && \
    tar -xzf openbabel-3.1.1.tar.gz && \
    cd openbabel-3.1.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make && make install && \
    cd ../.. && rm -rf openbabel-3.1.1 openbabel-3.1.1.tar.gz || { echo "ERROR: Open Babel installation failed"; exit 1; }
# Verify installations
RUN echo "Verifying fpocket..." && \
    which fpocket || { echo "ERROR: fpocket not found"; exit 1; } && \
    fpocket -h || { echo "ERROR: fpocket not executable"; exit 1; }
RUN echo "Verifying obabel..." && \
    which obabel || { echo "ERROR: obabel not found"; exit 1; } && \
    obabel -V || { echo "ERROR: obabel not executable"; exit 1; }
# Set PATH explicitly
ENV PATH="/usr/bin:/usr/local/bin:${PATH}"
# Debug PATH
RUN echo "PATH is: $PATH"
# Create static directory with write permissions
RUN mkdir -p /opt/render/project/src/static && chmod -R 777 /opt/render/project/src/static
# Create working directory with write permissions
RUN chmod -R 777 /app
# Update pip
RUN pip install --upgrade pip
# Install Python dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY . /app
EXPOSE 8080
CMD ["gunicorn", "--timeout", "600", "--workers", "1", "--threads", "4", "app:app"]