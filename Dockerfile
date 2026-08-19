# Use a slim Python 3.11 image
FROM python:3.11-slim

# Avoid interactive prompts and keep the image small
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# ChromaDB storage inside the container
RUN mkdir -p chroma_db data

# Expose Streamlit's default port
EXPOSE 8501

# Start the application
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.fileWatcherType=none"]
