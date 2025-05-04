#!/usr/bin/env python3
"""
Fine-tune DistilBERT on CSV logs stored in S3.
Works even on datasets<=2.18 because we bypass load_dataset(..., s3://…).
"""

import argparse, datetime, json, os, tarfile, tempfile, pathlib, boto3, s3fs
import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

LABELS = ["DEBUG", "WARN", "ERROR", "EXCEPTION"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ────────────────────────── CLI ──────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--bucket", required=True)              # S3 bucket name
ap.add_argument("--csv-prefix", default="classified_")  # key prefix
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--model-output", required=True)        # s3://bucket/path/
args = ap.parse_args()

# ─────────────────── collect CSVs from S3 ───────────────────
fs = s3fs.S3FileSystem()          # creds via IRSA
keys = fs.glob(f"{args.bucket}/{args.csv_prefix}*.csv")
if not keys:
    raise SystemExit("✗ No CSV files matched")

print(f"✓ {len(keys)} CSV files found, reading…")
datasets = []
for k in keys:
    # k looks like  "log-csv-bkt/classified_2025-04-30_12-15-02.csv"
    with fs.open(k, "rb") as f:
        for chunk in pd.read_csv(f, chunksize=50_000):
            chunk = chunk[chunk["tags"].str.contains("|".join(LABELS))]
            ds = Dataset.from_pandas(chunk, preserve_index=False)
            datasets.append(ds)
            del chunk

    # keep only the 4 target labels
    df = df[df["tags"].str.contains("|".join(LABELS))]
    datasets.append(Dataset.from_pandas(df, preserve_index=False))

dataset = concatenate_datasets(datasets).train_test_split(test_size=0.1, seed=42)
print(f"✓ Dataset built: {len(dataset['train'])} train / {len(dataset['test'])} val rows")

# ───────────────────── tokenise & label ─────────────────────
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def encode(batch):
    batch_enc = tok(batch["message"], truncation=True)
    batch_enc["labels"] = [
        LABEL2ID[tag.split("|")[0]] for tag in batch["tags"]
    ]
    return batch_enc

dataset = dataset.map(encode, batched=True, remove_columns=dataset["train"].column_names)
collator = DataCollatorWithPadding(tok)

# ───────────────────────── model ────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

training_args = TrainingArguments(
    output_dir          = "/tmp/out",
    num_train_epochs    = args.epochs,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    evaluation_strategy = "epoch",
    save_total_limit    = 1,
    logging_steps       = 50,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset = dataset["train"],
    eval_dataset  = dataset["test"],
    data_collator = collator,
)
trainer.train()

# ─────────────────── package & upload ───────────────────────
tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="bert_model_"))
model.save_pretrained(tmp_dir)
tok.save_pretrained(tmp_dir)
(tmp_dir / "label_map.json").write_text(json.dumps(ID2LABEL))

archive = tmp_dir.with_suffix(".tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    for f in tmp_dir.iterdir():
        tar.add(f, arcname=f.name)

dst = boto3.client("s3")
out = os.path.join(args.model_output.lstrip("/"), f"distilbert_{datetime.datetime.utcnow():%Y-%m-%d}.tar.gz")
bucket = args.model_output.replace("s3://", "").split("/", 1)[0]
dst.upload_file(str(archive), bucket, out)
print(f"✓ Model uploaded → s3://{bucket}/{out}")
