# ====================================================================================================
# frontend/openai_service/Dockerfile
# Dockerfile for the dedicated OpenAI Whisper service
# ====================================================================================================

# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for pydub and PyAudio
# ffmpeg is crucial for pydub
# PortAudio is a dependency for PyAudio (often needs system-level install)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port that Uvicorn will listen on
EXPOSE 8000

# Define the command to run your application
# This should match the start command you set on Railway
CMD ["uvicorn", "whisper_service:app", "--host", "0.0.0.0", "--port", "8000"]

# ====================================================================================================
