FROM python:3.10.0

# Set root user (if needed)
USER root

WORKDIR /app

# Copy files
COPY requirements.txt /app/
COPY Stock_Prediction.csv /app/
COPY final.py /app/

# Create cache directory and set permissions
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV HF_TOKEN=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

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "final.py"]
