FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .

EXPOSE 8080
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:8080", "serve:app"]
