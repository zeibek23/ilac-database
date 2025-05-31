# Use official Python runtime as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set PATH for Conda
ENV PATH="/opt/conda/bin:/opt/conda/envs/drugly_env/bin:${PATH}"

# Create Conda environment
RUN conda create -n drugly_env python=3.9 -y

# Install fpocket and Open Babel in Conda environment
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate drugly_env && \
    conda config --add channels conda-forge && \
    conda install -y fpocket openbabel=3.1.1"

# Pre-download NLTK data
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate drugly_env && \
    python -c 'import nltk; nltk.download(\"brown\"); nltk.download(\"punkt\")'"

# Verify installations during build
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate drugly_env && \
    which fpocket && fpocket -h && \
    which obabel && obabel -V" || { echo "ERROR: fpocket or obabel not found"; exit 1; }

# Install Python dependencies
COPY requirements.txt .
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate drugly_env && \
    /opt/conda/envs/drugly_env/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/conda/envs/drugly_env/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/conda/envs/drugly_env/bin/pip install --no-cache-dir gunicorn==22.0.0"

# Copy application code
COPY . .

# Create static directory with correct permissions
RUN mkdir -p /opt/render/project/src/static && \
    chmod -R 755 /opt/render/project/src/static

# Create app user and set permissions
RUN useradd -m appuser && \
    chown -R appuser:appuser /app /opt/render/project/src/static && \
    chmod -R u+rw /opt/conda/envs/drugly_env

# Switch to appuser
USER appuser

# Ensure /app is writable
RUN chmod -R u+rw /app

# Expose port (Render uses PORT environment variable)
EXPOSE 8000

# Command to run the application with explicit Conda activation
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate drugly_env && exec gunicorn -b 0.0.0.0:$PORT --timeout 1200 --workers 4 --threads 4 app:app"]