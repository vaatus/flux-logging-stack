#!/usr/bin/env python3
"""
Pull recent logs from Elasticsearch, tag them, export Prometheus metrics
(port 8000) and optionally upload each CSV snapshot to S3.
"""

import argparse, datetime, os, ssl, time, json
from typing import Dict, List, Tuple

import pandas as pd
import boto3, botocore.exceptions, pathlib
from elasticsearch import Elasticsearch
from prometheus_client import Counter, Gauge, start_http_server

# ── Prometheus metrics ─────────────────────────────────────────────────────────
RUN_OK  = Gauge("lc_last_run_success",  "1 = success, 0 = error")
RUN_SEC = Gauge("lc_last_run_seconds",  "Seconds taken by last run")
TAG_CNT = Counter(
    "lc_tag_total",
    "Classified tag occurrences",
    labelnames=["tag"]
)

# ── Tagging rules (regex-free, ultra-light) ────────────────────────────────────
def tag_log(msg: str) -> List[str]:
    l = msg.lower()
    tags = []
    if "error"     in l: tags.append("ERROR")
    if "exception" in l: tags.append("EXCEPTION")
    if "warn"      in l: tags.append("WARN")
    if "debug"     in l: tags.append("DEBUG")
    if not tags and "info" in l:
        tags.append("INFO")
    return tags

# ── Single ES query + in-memory classification ────────────────────────────────
def classify_once(
    es: Elasticsearch,
    index_pat: str,
    hours: int,
    namespace: str,
    cont_pat: str
) -> Tuple[Dict[str, int], int, List[dict]]:
    """Returns (tag_counts, n_rows, rows[])"""
    query = {
        "bool": {
            "filter": [
                {"term":     {"kubernetes.namespace_name.keyword": namespace}},
                {"wildcard": {"kubernetes.container_name.keyword": cont_pat}},
                {"range":    {"@timestamp": {"gte": f"now-{hours}h"}}}
            ]
        }
    }

    resp = es.search(index=index_pat, query=query, size=10_000)  # ES default cap
    rows: List[dict] = []
    counts: Dict[str, int] = {}

    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        msg = src.get("log", src.get("message", ""))
        tags = tag_log(msg)
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        rows.append({
            "@timestamp": src.get("@timestamp"),
            "message":    msg,
            "tags":       "|".join(tags)
        })

    return counts, len(rows), rows

# ── Elasticsearch client helper ────────────────────────────────────────────────
def make_es(host: str, user: str, passwd: str, ca_file: str) -> Elasticsearch:
    ctx = ssl.create_default_context(cafile=ca_file)
    return Elasticsearch(
        hosts=[host],
        basic_auth=(user, passwd),
        ssl_context=ctx,
        verify_certs=True,
        headers={"Accept": "application/json"}
    )

# ── CSV writer (+ optional S3 upload) ──────────────────────────────────────────
def write_csv(rows: List[dict], out_dir: str, s3_bucket: str | None) -> None:
    if not rows:                              # nothing to write
        print("• no rows → CSV skipped", flush=True)
        return

    ts   = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    name = f"classified_{ts}.csv"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"• CSV written → {path}", flush=True)

    if s3_bucket:
        try:
            boto3.client("s3").upload_file(path, s3_bucket, name)
            print(f"• CSV uploaded → s3://{s3_bucket}/{name}", flush=True)
        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            msg  = e.response["Error"]["Message"]
            print(f"✗ S3 upload failed ({code}): {msg}", flush=True)


# ── Main loop ──────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--es-host", required=True)
    p.add_argument("--es-user", required=True)
    p.add_argument("--es-pass", required=True)
    p.add_argument("--es-ca",   required=True)

    p.add_argument("--index-pattern",      default="fluent-bit*")
    p.add_argument("--namespace",          default="flux-system")
    p.add_argument("--container-wildcard", default="random-logger*")
    p.add_argument("--time-range", type=int, default=1,   help="Hours to look back")
    p.add_argument("--interval",   type=int, default=300, help="Seconds between runs")
    p.add_argument("--csv-dir",    default="/output")
    p.add_argument("--s3-bucket",  default=os.getenv("CSV_S3_BUCKET", ""))

    args = p.parse_args()
    es   = make_es(args.es_host, args.es_user, args.es_pass, args.es_ca)

    # Prometheus HTTP
    start_http_server(8000)
    print("Serving Prometheus metrics on :8000/metrics")

    while True:
        t0 = time.time()
        try:
            counts, total, rows = classify_once(
                es, args.index_pattern, args.time_range,
                args.namespace, args.container_wildcard)

            write_csv(rows, args.csv_dir, args.s3_bucket)

            dt = time.time() - t0
            RUN_OK.set(1)
            RUN_SEC.set(dt)

            for tag, val in counts.items():
                TAG_CNT.labels(tag).inc(val)

            print(f"✓ {total} logs classified in {dt:.2f}s {counts}", flush=True)
        except Exception as exc:
            RUN_OK.set(0)
            print("✗ classifier error:", exc, flush=True)

        time.sleep(args.interval)

if __name__ == "__main__":
    main()
