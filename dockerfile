FROM python:3.9
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libeigen3-dev \
    zlib1g-dev \
    libboost-dev \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://downloads.sourceforge.net/project/fpocket/fpocket-3.1.5.tar.gz && \
    tar -xzf fpocket-3.1.5.tar.gz && \
    cd fpocket-3.1.5 && \
    make && make install && \
    cd .. && rm -rf fpocket-3.1.5 fpocket-3.1.5.tar.gz
RUN wget https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1.tar.gz && \
    tar -xzf openbabel-3.1.1.tar.gz && \
    cd openbabel-3.1.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make && make install && \
    cd ../.. && rm -rf openbabel-3.1.1 openbabel-3.1.1.tar.gz
ENV PATH="/usr/local/bin:${PATH}"
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
CMD ["gunicorn", "--timeout", "600", "--workers", "1", "--threads", "4", "app:app"]