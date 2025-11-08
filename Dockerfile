# Use official Python image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt

# Expose Flask port
EXPOSE 8080

# Run the Flask app
CMD ["python3", "app.py"]
