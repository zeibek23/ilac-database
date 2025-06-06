# Python 3.11 slim yerine Ubuntu tabanlı bir imaj kullanalım (daha güvenilir bağımlılık yönetimi için)
FROM ubuntu:22.04

# Ortam değişkenlerini ayarla
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

# Sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    unzip \
    python3 \
    python3-pip \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

# Fpocket2'yi kaynaktan kur
RUN wget https://sourceforge.net/projects/fpocket/files/fpocket2.tar.gz/download -O fpocket2.tar.gz && \
    tar -xzf fpocket2.tar.gz && \
    mv fpocket* fpocket2 && \
    cd fpocket2 && \
    make && make install && \
    cd .. && \
    rm -rf fpocket2 fpocket2.tar.gz

# Binary'lerin PATH'te olduğunu doğrula
RUN which fpocket && fpocket -h && \
    which obabel && obabel -V || { echo "ERROR: fpocket or obabel not found"; exit 1; }

# Çalışma dizinini oluştur
WORKDIR /app

# Python bağımlılıklarını kur
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir gunicorn==22.0.0

# NLTK verilerini indir
RUN python3 -c 'import nltk; nltk.download("brown"); nltk.download("punkt")'

# Uygulama dosyalarını kopyala
COPY . .

# Statik dizini oluştur ve izinleri ayarla
RUN mkdir -p /app/static && \
    chmod -R 755 /app/static

# Uygulama kullanıcısı oluştur ve izinleri ayarla
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /app/static

# appuser'a geç
USER appuser

# /app dizininin yazılabilir olduğundan emin ol
RUN chmod -R u+rw /app

# Portu aç
EXPOSE 8000

# Runtime'da PATH ve binary'leri kontrol et, ardından gunicorn'u başlat
CMD ["/bin/bash", "-c", "echo 'Runtime PATH: $PATH' && which fpocket && which obabel && exec gunicorn -b 0.0.0.0:$PORT --timeout 1200 --workers 4 --threads 4 app:app"]