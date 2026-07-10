# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy .env file (important for your API key)
COPY .env .

# Copy your main code
COPY main_code.py .

# Set proper permissions for .env file (security)
RUN chmod 600 .env

# Command to run when container starts
CMD ["python", "main_code.py"]