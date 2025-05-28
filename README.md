# Automated Log Classification Pipeline

## I. Introduction

Modern cloud-native applications are built as microservices—small, independently deployable components that emit large volumes of unstructured log data. While traditional observability stacks (Fluent Bit → Elasticsearch → Kibana/Grafana) let engineers search and filter logs, they depend on pre-configured queries and alert rules, making it hard to catch novel errors in real time.  

This project delivers a fully automated, GitOps-managed pipeline that augments the classic stack with machine learning. Container logs are collected by **Fluent Bit** [^1] and indexed in **Elasticsearch** [^2]. A lightweight Python service periodically tags new entries via extensible regex rules, writes labeled CSV snapshots to **Amazon S3** [^3], and exposes Prometheus metrics [^4]. Each night a Kubernetes **CronJob** fine-tunes a **DistilBERT** model [^5] on those snapshots, pushes the updated checkpoint back to S3, and a Flask+Gunicorn API serves real-time `/predict` inferences. Infrastructure and deployments are entirely defined in Git (Flux v2 + HelmRelease + Kustomize), secured via AWS IAM Roles for Service Accounts (IRSA) [^6], and observed with **Prometheus** [^7] and **Grafana** [^8].

## II. State at Semester Start

At the beginning of the term:
- No existing infrastructure or codebase  
- Required tooling learned from scratch: WSL2, Docker, `kubectl`, `eksctl`, AWS CLI, Flux v2  

## III. Architecture Overview

![Fig. 1 – End-to-end architecture of the GitOps-managed logging & ML pipeline](![image](https://github-production-user-asset-6210df.s3.amazonaws.com/118371927/448576174-ae1b0306-7aa5-4c1a-8fce-322e8c1f3187.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250528%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250528T203423Z&X-Amz-Expires=300&X-Amz-Signature=0dbe8f936bbb0b9f7fd67d52e91e6726fa08c5f0eeda4eedb6e26d027c3d5ff1&X-Amz-SignedHeaders=host)

1. **Log Generation**  
   Synthetic or real container logs →  
2. **Collection**  
   Fluent Bit DaemonSet tails `/var/log/containers/*`, enriches metadata, ships via TLS  
3. **Indexing & Exploration**  
   Elasticsearch single-node PVC cluster → Kibana visual queries  
4. **Regex-Based Tagging**  
   `log-classifier` Deployment pulls last 10 000 logs, assigns tags (DEBUG/INFO/WARN/ERROR/EXCEPTION), writes CSV to S3, exports Prometheus metrics  
5. **Nightly Fine-Tuning**  
   `log-bert-train` CronJob streams CSVs from S3, balances classes, fine-tunes DistilBERT for 2 epochs, uploads new model archive to S3  
6. **Real-Time Serving**  
   `log-bert-serve` Deployment loads latest model on startup, serves `/predict` via Flask+Gunicorn (CPU or GPU), probes `/healthz`  
7. **Observability**  
   kube-prometheus-stack scrapes metrics from each component; Grafana dashboards surface resource usage, classification health, training performance  
8. **GitOps & Security**  
   Flux v2 reconciles all YAML every 5 min; IRSA-scoped IAM roles grant least-privilege S3 access; TLS and non-root containers enforce cluster security  

## IV. Installation & Usage

1. **Bootstrap Flux & EKS**  
   ```bash
   eksctl create cluster ... --enable-iam-roles-for-service-accounts
   flux bootstrap github \
     --owner=<you> --repository=<repo> --path=clusters/eks-cluster/flux-system
2. Deploy Services
   ```bash
   cd clusters/eks-cluster/flux-system
   git add .
   git commit -m "Initial pipeline"
   git push
   # Flux will apply all HelmRelease and Kustomization manifests
   ```
4. Access Dashboards

  Kibana → via LoadBalancer on port 5601

  Grafana → via NLB on port 80

4. Invoke Prediction
   ```bash
     kubectl port-forward svc/log-bert-serve 80 -n flux-system
     curl -XPOST localhost/predict \
       -H 'Content-Type: application/json' \
       -d '{"message":"An error occurred"}'
   ```

## V. Components
- Fluent Bit: Lightweight log forwarder, tails files, enriches metadata, TLS output.

- Elasticsearch: Distributed full-text search engine, stores indices on EBS PVC.

- Amazon S3: Durable object storage for CSV snapshots & model archives.

- Prometheus: Time-series metrics database, scrapes exporters & services.

- Grafana: Interactive dashboard UI for Prometheus data.

- Flux v2: GitOps operator, reconciles Kubernetes manifests from Git every 5 minutes.

- DistilBERT: Compact Transformer model, fine-tuned nightly for log severity [^5].

- Flask: Minimal Python web framework for REST APIs.

- Gunicorn: WSGI HTTP server to run Flask apps with multiple workers.

- AWS IAM IRSA: Secure pod-level AWS permissions via IAM Roles for Service Accounts.


