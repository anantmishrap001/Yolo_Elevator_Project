# Use a base image with Python (and optionally GPU support)
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# (Optional) expose a port if you run a web service
# e.g. if your Backend.py serves an API on port 5000
EXPOSE 5000

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Command to run when container starts
CMD ["python", "Backend.py"]
