#!/bin/sh
set -eu

tf_dir=${1:?Terraform directory is required}
context=${2:?Kubernetes context is required}
local_port=${3:?Local tunnel port is required}

cluster_name=$(terraform -chdir="$tf_dir" output -raw cluster_name)
region=$(terraform -chdir="$tf_dir" output -raw aws_region)
endpoint_host=$(terraform -chdir="$tf_dir" output -raw cluster_endpoint_hostname)

aws eks update-kubeconfig \
  --name "$cluster_name" \
  --region "$region" \
  --alias "$context"

cluster_ref=$(kubectl config view --raw \
  -o jsonpath="{.contexts[?(@.name==\"$context\")].context.cluster}")
test -n "$cluster_ref"

kubectl config set-cluster "$cluster_ref" \
  --server="https://127.0.0.1:$local_port" \
  --tls-server-name="$endpoint_host"
