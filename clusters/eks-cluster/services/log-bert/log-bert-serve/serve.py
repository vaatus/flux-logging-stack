"""
Tiny REST API for log-level classification.

Startup:
• Downloads a model tarball (MODEL_TAR) from S3_BUCKET
• Extracts it, loads tokenizer + model + label_map.json
• Supports CPU or CUDA transparently

Endpoints:
• POST /predict   { "message": "text" | ["text1", "text2"] }
                  → { "tag": "ERROR", "confidence": 0.97 } | [...]
•   GET /healthz  → 200 OK     (K8s liveness/readiness)

Environment:
  S3_BUCKET   = log-csv-bkt
  MODEL_TAR   = models/bert-tiny_2025-05-04.tar.gz
  PORT        = 8080   (optional)
"""

import os, tarfile, json, logging, boto3, torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("serve")

# ─────────────────────── config ───────────────────────────
S3_BUCKET = os.environ["S3_BUCKET"]
MODEL_TAR = os.environ["MODEL_TAR"]       # key, e.g. models/bert-tiny_2025-05-04.tar.gz
PORT      = int(os.getenv("PORT", "8080"))

TMP_TAR = "/tmp/model.tar.gz"
TMP_DIR = "/tmp/model"

# ─────────────────── download & extract ───────────────────
log.info("Downloading s3://%s/%s …", S3_BUCKET, MODEL_TAR)
boto3.client("s3").download_file(S3_BUCKET, MODEL_TAR, TMP_TAR)

with tarfile.open(TMP_TAR) as tar:
    tar.extractall(path=TMP_DIR)

with open(os.path.join(TMP_DIR, "label_map.json")) as f:
    ID2LABEL = {int(k): v for k, v in json.load(f).items()}

tokenizer = AutoTokenizer.from_pretrained(TMP_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(TMP_DIR)
model.eval()                       # no dropout
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

log.info("Model loaded (%s, %d labels)", DEVICE, len(ID2LABEL))

# ───────────────────────── API ────────────────────────────
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "JSON body must contain 'message'"}), 400

    msgs = data["message"]
    if isinstance(msgs, str):
        msgs = [msgs]
    if not isinstance(msgs, list):
        return jsonify({"error": "'message' must be string or list"}), 400

    enc = tokenizer(msgs, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    results = [
        {"tag": ID2LABEL[idx_i.item()], "confidence": round(conf_i.item(), 5)}
        for idx_i, conf_i in zip(idx, conf)
    ]
    return jsonify(results[0] if len(results) == 1 else results)

@app.route("/healthz")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
