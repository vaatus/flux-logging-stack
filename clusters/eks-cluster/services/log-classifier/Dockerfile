FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY classify_logs.py .

EXPOSE 8000
VOLUME ["/output"]
ENTRYPOINT ["python", "/app/classify_logs.py"]
