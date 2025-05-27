FROM python:3.9
WORKDIR /app
# Install fpocket and Open Babel
RUN apt-get update && apt-get install -y fpocket openbabel
# Copy your app code
COPY . /app
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Expose port
EXPOSE 8080
# Run Flask app with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]