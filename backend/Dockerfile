FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy download script
COPY download_models.py ./

# Download model files during build
RUN python download_models.py

# Debug: Check file sizes
RUN ls -lh checkpoint.pt train.txt
RUN echo "Checkpoint size: $(wc -c < checkpoint.pt) bytes"
RUN echo "Train text size: $(wc -c < train.txt) bytes"

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
