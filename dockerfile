# Use official Python runtime as the base image
FROM python:3.9

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
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda init bash

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

# Verify installations
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate drugly_env && \
    which fpocket && fpocket -h && \
    which obabel && obabel -V" || { echo "ERROR: fpocket or obabel not found"; exit 1; }

# Set permissions for Conda environment
RUN chown -R 1000:1000 /opt/conda/envs/drugly_env && \
    chmod -R u+rw /opt/conda/envs/drugly_env

# Create static directory with correct permissions
RUN mkdir -p /opt/render/project/src/static && \
    chown -R 1000:1000 /opt/render/project/src/static && \
    chmod -R 755 /opt/render/project/src/static

# Create app user and set permissions
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Install Python dependencies
COPY requirements.txt .
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate drugly_env && \
    /opt/conda/envs/drugly_env/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/conda/envs/drugly_env/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/conda/envs/drugly_env/bin/pip install --no-cache-dir gunicorn==22.0.0"

# Copy application code
COPY --chown=appuser:appuser . .

# Ensure /app is writable
RUN chown -R appuser:appuser /app

# Expose port (Render uses PORT environment variable, but set for clarity)
EXPOSE 8000

# Command to run the application
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate drugly_env && gunicorn -b 0.0.0.0:$PORT --timeout 1200 --workers 4 --threads 4 app:app"]