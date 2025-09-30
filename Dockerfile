# ====================================================================================================
# frontend/openai_service/Dockerfile (UPDATED to correct CMD instruction)
# Dockerfile for the dedicated OpenAI Whisper service
# ====================================================================================================

# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for pydub and PyAudio
# ffmpeg is crucial for pydub
# libportaudio2 and portaudio19-dev are for PyAudio
# build-essential provides gcc and other build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
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
# MODIFIED: Use shell form for CMD to allow $PORT environment variable expansion
CMD ["python", "whisper_service.py"]

# ====================================================================================================
