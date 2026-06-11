# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir shap

# Copy all project files
COPY . .

# Create required directories
RUN mkdir -p models reports data

# Train models if not present
RUN python train_models.py

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]