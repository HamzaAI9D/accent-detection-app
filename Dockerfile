# Step 1: Use a base image
FROM python:3.12-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy application code
COPY . /app/

# Step 6: Expose port for the app to run on
EXPOSE 8501

# Step 7: Command to run the app
CMD ["streamlit", "run", "app.py"]
