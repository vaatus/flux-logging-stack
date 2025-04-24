#!/usr/bin/env python3
"""
Continuously pulls recent logs from Elasticsearch, tags them, exports
Prometheus metrics (port 8000) and writes a CSV snapshot to /output/.
"""

import argparse, datetime, os, ssl, time
from typing import Dict, List, Tuple

import pandas as pd
import boto3, pathlib, botocore.exceptions
from elasticsearch import Elasticsearch
from prometheus_client import Counter, Gauge, start_http_server

# - Prometheus metrics
RUN_OK  = Gauge ('lc_last_run_success', '1=success,0=error')
RUN_SEC = Gauge ('lc_last_run_seconds', 'Seconds in last run')
TAG_CNT = Counter('lc_tag_total',       'Classified tags', ['tag'])

# - tagging rules 
def tag_log(msg: str) -> List[str]:
    l   = msg.lower()
    out = []
    if 'error'     in l: out.append('ERROR')
    if 'exception' in l: out.append('EXCEPTION')
    if 'warn'      in l: out.append('WARN')
    if 'debug'     in l: out.append('DEBUG')
    if not out and 'info' in l:
        out.append('INFO')
    return out

# - single ES query + classification
def classify_once(
    es: Elasticsearch,
    index_pat: str,
    hours: int,
    namespace: str,
    cont_pat: str
) -> Tuple[Dict[str, int], int, List[dict]]:

    query = {
      "bool": {
        "filter": [
          { "term":     {"kubernetes.namespace_name.keyword": namespace} },
          { "wildcard": {"kubernetes.container_name.keyword": cont_pat} },
          { "range":    {"@timestamp": {"gte": f"now-{hours}h"}} }
        ]
      }
    }

    resp = es.search(index=index_pat, query=query, size=10_000)

    counts: Dict[str, int] = {}
    rows:   List[dict]     = []

    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        msg = src.get("log", src.get("message", ""))
        tags = tag_log(msg)
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        rows.append({
            "@timestamp": src.get("@timestamp"),
            "message":    msg,
            "tags":       '|'.join(tags)
        })

    return counts, len(rows), rows

# - ES client helper 
def make_es(host: str, user: str, passwd: str, ca_file: str) -> Elasticsearch:
    ctx = ssl.create_default_context(cafile=ca_file)
    return Elasticsearch(
        hosts=[host],
        basic_auth=(user, passwd),
        ssl_context=ctx,
        verify_certs=True,
        headers={"Accept": "application/json"}
    )

# - CSV writer + uploader
def write_csv(rows: List[dict], out_dir: str, s3_bucket: str | None) -> None:
    if not rows:
        return

    ts   = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"classified_{ts}.csv"
    path = os.path.join(out_dir, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"• CSV written: {path}", flush=True)

    if not s3_bucket:
        return

    try:
        s3 = boto3.client('s3')
        s3.upload_file(path, s3_bucket, name)
        print(f"• CSV uploaded → s3://{s3_bucket}/{name}", flush=True)
    except botocore.exceptions.BotoCoreError as err:
        print("✗ S3 upload failed:", err, flush=True)


# - main loop 
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--es-host', required=True)
    p.add_argument('--es-user', required=True)
    p.add_argument('--es-pass', required=True)
    p.add_argument('--es-ca',   required=True)

    p.add_argument('--index-pattern',      default='fluent-bit*')
    p.add_argument('--namespace',          default='flux-system')
    p.add_argument('--container-wildcard', default='random-logger*')
    p.add_argument('--time-range', type=int, default=1,
                   help='Hours to look back')
    p.add_argument('--interval',   type=int, default=300,
                   help='Seconds between runs')
    p.add_argument('--csv-dir',    default='/output',
                   help='Where CSVs are written (mounted volume)')
    p.add_argument('--s3-bucket', default=os.getenv("CSV_S3_BUCKET", ""),
               help='If set, upload each CSV to this S3 bucket')
    
    args = p.parse_args()
    write_csv(rows, args.csv_dir, args.s3_bucket)
    es = make_es(args.es_host, args.es_user, args.es_pass, args.es_ca)
    start_http_server(8000)
    print("Serving Prometheus metrics on :8000/metrics")

    while True:
        start = time.time()
        try:
            counts, total, rows = classify_once(
                es, args.index_pattern, args.time_range,
                args.namespace, args.container_wildcard)

            write_csv(rows, args.csv_dir)

            RUN_OK.set(1)
            RUN_SEC.set(time.time() - start)
            for tag, val in counts.items():
                TAG_CNT.labels(tag).inc(val)

            print(f"✓ {total} logs classified in {RUN_SEC._value.get():.2f}s {counts}",
                  flush=True)
        except Exception as exc:
            RUN_OK.set(0)
            print("✗ classifier error:", exc, flush=True)

        time.sleep(args.interval)

if __name__ == '__main__':
    main()
