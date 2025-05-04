#!/usr/bin/env python3
"""
Fast fine-tune of a tiny BERT on log CSVs stored in S3.

  • Reads every CSV in 50 k-row chunks
  • Keeps at most --max-per-class unique messages per label
  • Uses bert-tiny for speed (change MODEL_NAME to distilbert-base-uncased if
    you later need more accuracy)
  • Auto-detects GPU → fp16 + bigger batch
"""

import argparse, datetime, json, os, tarfile, tempfile, pathlib, random
from collections import defaultdict

import boto3, pandas as pd, s3fs, torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

LABELS   = ["DEBUG", "WARN", "ERROR", "EXCEPTION"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ───────────────────────── CLI ────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--bucket",       required=True)
p.add_argument("--csv-prefix",   default="classified_")
p.add_argument("--model-output", required=True)          # s3://bucket/path/
p.add_argument("--epochs",       type=int, default=1)
p.add_argument("--max-per-class",type=int, default=5_000)
args = p.parse_args()

# ─────────────── 1. Stream & sample CSVs ──────────────────
fs   = s3fs.S3FileSystem()          # IRSA creds
keys = fs.glob(f"{args.bucket}/{args.csv_prefix}*.csv")
if not keys:
    raise SystemExit("✗ No CSV files matched")

print(f"✓ {len(keys)} CSV files found, sampling up to "
      f"{args.max_per_class} unique messages per label")

rows, seen = [], set()
cnt = defaultdict(int)

for k in keys:
    with fs.open(k, "rb") as f:
        for chunk in pd.read_csv(f, chunksize=50_000):
            for _, row in chunk.iterrows():
                tag = row["tags"].split("|")[0]
                if tag not in LABEL2ID:
                    continue
                msg = str(row["message"])
                key = (tag, msg)
                if key in seen or cnt[tag] >= args.max_per_class:
                    continue
                seen.add(key)
                cnt[tag] += 1
                rows.append({"message": msg, "tags": tag})
            del chunk
    # stop early if we already hit the cap for every class
    if all(cnt[l] >= args.max_per_class for l in LABELS):
        break

print("✓ Sampled rows:", len(rows))
dataset = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)

# ───────────── 2. Tokenise & collate ──────────────────────
MODEL_NAME = "prajjwal1/bert-tiny"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(batch):
    out = tok(batch["message"], truncation=True)
    out["labels"] = [LABEL2ID[b] for b in batch["tags"]]
    return out

dataset = dataset.map(encode, batched=True,
                      remove_columns=dataset["train"].column_names)
collator = DataCollatorWithPadding(tok)

# ───────────── 3. Model & training args ───────────────────
has_gpu   = torch.cuda.is_available()
batch_sz  = 32 if has_gpu else 8

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = len(LABELS),
    id2label   = ID2LABEL,
    label2id   = LABEL2ID,
)
model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir               = "/tmp/out",
    num_train_epochs         = args.epochs,
    per_device_train_batch_size = batch_sz,
    per_device_eval_batch_size  = batch_sz,
    fp16                     = has_gpu,
    evaluation_strategy      = "epoch",
    logging_steps            = 20,
    save_total_limit         = 1,
)

print(f"▶ Training on {'GPU' if has_gpu else 'CPU'} "
      f"batch={batch_sz} epochs={args.epochs}")
Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
).train()

# ───────────── 4. Package & upload ────────────────────────
tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="bert_model_"))
model.save_pretrained(tmp_dir)
tok.save_pretrained(tmp_dir)
(tmp_dir / "label_map.json").write_text(json.dumps(ID2LABEL))

archive = tmp_dir.with_suffix(".tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    for f in tmp_dir.iterdir():
        tar.add(f, arcname=f.name)

s3  = boto3.client("s3")
dst = args.model_output.replace("s3://", "")
bucket, key_prefix = dst.split("/", 1)
key = f"{key_prefix.rstrip('/')}/bert-tiny_{datetime.datetime.utcnow():%Y-%m-%d}.tar.gz"
s3.upload_file(str(archive), bucket, key)
print(f"✓ Model uploaded → s3://{bucket}/{key}")
