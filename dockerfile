FROM python:3.9
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libeigen3-dev \
    zlib1g-dev \
    libboost-dev \
    && rm -rf /var/lib/apt/lists/*

# Install fpocket from source
RUN wget https://downloads.sourceforge.net/project/fpocket/fpocket-3.1.5.tar.gz && \
    tar -xzf fpocket-3.1.5.tar.gz && \
    cd fpocket-3.1.5 && \
    make && make install && \
    cd .. && rm -rf fpocket-3.1.5 fpocket-3.1.5.tar.gz

# Install Open Babel from source
RUN wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1.tar.gz && \
    tar -xzf openbabel-3.1.1.tar.gz && \
    cd openbabel-3.1.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make && make install && \
    cd ../.. && rm -rf openbabel-3.1.1 openbabel-3.1.1.tar.gz

# Update PATH to include Open Babel and fpocket
ENV PATH="/usr/local/bin:${PATH}"

# Copy app code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Run Flask app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]