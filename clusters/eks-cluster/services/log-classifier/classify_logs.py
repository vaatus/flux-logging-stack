#!/usr/bin/env python3
"""
Continuously pulls recent logs from Elasticsearch, tags them, and exposes
Prometheus metrics on http://0.0.0.0:8000/metrics

Metrics:
  lc_tag_total{tag="ERROR"}   ever-increasing counter per tag
  lc_last_run_success         1 if last run OK, 0 on error
  lc_last_run_seconds         duration of last ES query + processing
"""

import argparse, ssl, time
from typing import Dict, List

import pandas as pd               #imported for future CSV output
from elasticsearch import Elasticsearch
from prometheus_client import Counter, Gauge, start_http_server
from typing import Dict, Tuple, List

# Prometheus metrics
RUN_OK  = Gauge ('lc_last_run_success', '1=success,0=error')
RUN_SEC = Gauge ('lc_last_run_seconds', 'Seconds in last run')
TAG_CNT = Counter('lc_tag_total',       'Classified tags', ['tag'])

# tagging logic
def tag_log(msg: str) -> List[str]:
    l = msg.lower()
    tags = []
    if 'error' in l:      tags.append('ERROR')
    if 'exception' in l:  tags.append('EXCEPTION')
    if 'warn' in l:       tags.append('WARN')
    if 'debug' in l:      tags.append('DEBUG')
    if not tags and 'info' in l:
        tags.append('INFO')
    return tags

def classify_once(es: Elasticsearch,
                  index_pat: str,
                  hours: int,
                  namespace: str,
                  cont_pat: str) -> Tuple[Dict[str, int], int]:

    query = {
      "bool": {
        "filter": [
          { "term": {"kubernetes.namespace_name.keyword": namespace} },
          { "wildcard": {"kubernetes.container_name.keyword": cont_pat} },
          { "range": {"@timestamp": {"gte": f"now-{hours}h"}} }
        ]
      }
    }

    resp = es.search(index=index_pat, query=query, size=10_000)
    counts: Dict[str, int] = {}

    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        msg = src.get("log", src.get("message", ""))
        for tag in tag_log(msg):
            counts[tag] = counts.get(tag, 0) + 1

    return counts, len(resp["hits"]["hits"])


# ES client with TLS
def make_es(host: str, user: str, passwd: str, ca_file: str) -> Elasticsearch:
    ctx = ssl.create_default_context(cafile=ca_file)
    return Elasticsearch(
        hosts=[host],
        basic_auth=(user, passwd),
        ssl_context=ctx,
        verify_certs=True,
        # forces simple JSON if a newer client is ever used
        headers={"Accept": "application/json"}
    )

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--es-host',  required=True)
    p.add_argument('--es-user',  required=True)
    p.add_argument('--es-pass',  required=True)
    p.add_argument('--es-ca',    required=True)
    p.add_argument('--index-pattern', default='fluent-bit*',
                   help='Index pattern or alias to search')
    p.add_argument('--namespace', default='flux-system',
                   help='Kubernetes namespace to filter')
    p.add_argument('--container-wildcard', default='random-logger*',
                   help='Wildcard for kubernetes.container_name.keyword')
    p.add_argument('--time-range', type=int, default=1,
                   help='Look-back window in hours')
    p.add_argument('--interval',   type=int, default=300,
                   help='Seconds between runs')
    args = p.parse_args()

    es = make_es(args.es_host, args.es_user, args.es_pass, args.es_ca)
    start_http_server(8000)
    print("Serving Prometheus metrics on :8000/metrics")

    while True:
        t0 = time.time()
        try:
            counts, total = classify_once(
                es, args.index_pattern, args.time_range,
                args.namespace, args.container_wildcard)

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


if __name__ == '__main__':
    main()
