# Base image with Python 3.10
FROM python:3.10-slim

# Install system dependencies for ffmpeg (used by pydub)
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "appcore:app", "--host", "0.0.0.0", "--port", "8000"]
