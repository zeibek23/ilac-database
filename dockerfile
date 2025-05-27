FROM python:3.9
WORKDIR /app
# Update package lists and install fpocket and openbabel
RUN apt-get update && apt-get install -y \
    fpocket \
    openbabel \
    && rm -rf /var/lib/apt/lists/*
# Copy app code
COPY . /app
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose port
EXPOSE 8080
# Run Flask app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]