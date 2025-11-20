# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install necessary system dependencies (needed by Pillow)
RUN apt-get update && apt-get install -y libpq-dev gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (includes Pillow)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script and run it to create the model (savedmodel.pth)
COPY train.py .
RUN python train.py 

# Copy the web application
COPY app.py .

EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]