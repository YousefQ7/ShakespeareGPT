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

# Copy model files (these will be mounted or copied separately)
COPY checkpoint.pt ./
COPY train.txt ./

# Expose port (will be set by Railway)
EXPOSE $PORT

# Run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
