#!/bin/sh
set -eu

tf_dir=${1:?Terraform directory is required}
local_port=${2:?Local tunnel port is required}

target=$(terraform -chdir="$tf_dir" output -raw admin_instance_id)
endpoint_host=$(terraform -chdir="$tf_dir" output -raw cluster_endpoint_hostname)

exec aws ssm start-session \
  --target "$target" \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters "{\"host\":[\"$endpoint_host\"],\"portNumber\":[\"443\"],\"localPortNumber\":[\"$local_port\"]}"
