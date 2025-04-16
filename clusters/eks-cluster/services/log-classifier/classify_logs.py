#!/usr/bin/env python3
import argparse, ssl, time
import pandas as pd
from elasticsearch import Elasticsearch
from prometheus_client import Counter, Gauge, start_http_server

RUN_OK   = Gauge ('lc_last_run_success',  '1=success,0=error')
RUN_SEC  = Gauge ('lc_last_run_seconds',  'Seconds in last run')
TAG_CNT  = Counter('lc_tag_total',        'Classified tags', ['tag'])

def tag_log(msg:str)->list:
    l=msg.lower(); tags=[]
    if 'error' in l: tags.append('ERROR')
    if 'exception' in l: tags.append('EXCEPTION')
    if 'warn' in l: tags.append('WARN')
    if 'debug' in l: tags.append('DEBUG')
    if not tags and 'info' in l: tags.append('INFO')
    return tags

def classify_once(es,index_pat,hours):
    q={"bool":{"filter":[
         {"term":{"kubernetes.container_name":"random-logger"}},
         {"range":{"@timestamp":{"gte":f"now-{hours}h"}}}
    ]}}
    resp=es.search(index=index_pat,query=q,size=10000)
    counts={}
    for hit in resp["hits"]["hits"]:
        msg=hit["_source"].get("log",hit["_source"].get("message",""))
        for t in tag_log(msg):
            counts[t]=counts.get(t,0)+1
    return counts,len(resp["hits"]["hits"])

def make_es(host,user,passwd,ca_file):
    ctx=ssl.create_default_context(cafile=ca_file)
    return Elasticsearch(
        hosts=[host],
        basic_auth=(user,passwd),
        ssl_context=ctx,
        verify_certs=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--es-host',required=True)
    p.add_argument('--es-user',required=True)
    p.add_argument('--es-pass',required=True)
    p.add_argument('--es-ca',required=True)
    p.add_argument('--index-pattern',default='fluent-bit*')
    p.add_argument('--time-range',type=int,default=1)
    p.add_argument('--interval',type=int,default=300)
    args=p.parse_args()

    es=make_es(args.es_host,args.es_user,args.es_pass,args.es_ca)
    start_http_server(8000)

    while True:
        t0=time.time()
        try:
            counts,total = classify_once(es,args.index_pattern,args.time_range)
            dt=time.time()-t0
            RUN_OK.set(1); RUN_SEC.set(dt)
            for tag,val in counts.items():
                TAG_CNT.labels(tag).inc(val)
            print(f"✓ {total} logs classified in {dt:.2f}s {counts}",flush=True)
        except Exception as e:
            RUN_OK.set(0)
            print("✗ classifier error:",e,flush=True)
        time.sleep(args.interval)

if __name__=='__main__':
    main()
