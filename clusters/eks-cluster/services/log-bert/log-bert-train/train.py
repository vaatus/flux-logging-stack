import os, argparse, s3fs, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True)
parser.add_argument("--csv-prefix", default="classified_")
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

# 1. load CSVs straight from S3
fs = s3fs.S3FileSystem()
csv_paths = fs.glob(f"{args.bucket}/{args.csv_prefix}*.csv")
dataset   = load_dataset(
              "csv",
              data_files=csv_paths,
              split="train",
              fs=fs)

# split train/validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# 2. tokenise
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def encode(batch): return tok(batch["message"], truncation=True)
dataset = dataset.map(encode, batched=True)

# 3. label map
labels = ["DEBUG","INFO","WARN","ERROR","EXCEPTION"]
id2l   = dict(enumerate(labels))
l2id   = {v:k for k,v in id2l.items()}
dataset = dataset.map(lambda b: {"label": [l2id.get(t.split('|')[0],0)
                                  for t in b["tags"]]})

model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(labels))

# 4. minimal training
args_tr = TrainingArguments(
    output_dir      = "/tmp/out",
    num_train_epochs= args.epochs,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_total_limit=1,
    push_to_hub=False)

trainer = Trainer(model, args_tr,
                  train_dataset=dataset["train"],
                  eval_dataset =dataset["test"])
trainer.train()
trainer.save_pretrained("/tmp/model")

# 5. upload artefacts
import boto3, pathlib, tarfile
s3 = boto3.client("s3")
tar_path="/tmp/model.tar.gz"
with tarfile.open(tar_path,"w:gz") as tar:
    for f in pathlib.Path("/tmp/model").glob("*"):
        tar.add(f, arcname=f.name)
s3.upload_file(tar_path, args.bucket, "models/distilbert.tar.gz")
print("Model uploaded.")