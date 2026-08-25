# SmolLM stalled-rank/XID recovery experiment — 2026-08-24

## Result

The AWS gate passed. A four-rank SmolLM3 LoRA fine-tune stopped making progress when rank 3's training worker was paused while its Kubernetes pod remained `Running`. Synthetic XID 79 telemetry then caused the GPU health controller to isolate the node and evict the stopped pod. JobSet recreated the DDP group away from that node, all four ranks restored exactly the stable EFS checkpoint, and CUDA/NCCL training advanced again.

This is a deterministic composite failure, not a manufactured physical XID. `SIGSTOP` supplies the application-visible stall; DCGM field injection supplies the accelerator-health observation. A natural XID 79 would cause both effects from one hardware/driver event.

## Workload and environment

- Private Amazon EKS 1.34 cluster in `us-east-1`.
- Six on-demand `g6.xlarge` workers with one NVIDIA L4 each: four active ranks and two replacement nodes.
- Kubeflow Trainer 2.2.1, JobSet, NVIDIA GPU Operator, and the repository's GPU node health controller.
- PyTorch DDP over NCCL with BF16, one process and GPU per node.
- `HuggingFaceTB/SmolLM3-3B` revision `a07cc9a04f16550a088caea529712d1d335b0ac1` and SmolTalk `smol-magpie-ultra` revision `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`.
- LoRA rank 16, sequence length 1024, gradient accumulation 4, and checkpoints every five optimizer steps on encrypted EFS.

The checkpoint includes adapter weights, optimizer and scheduler state, mixed-precision scaler state, every rank's Python/PyTorch/CUDA RNG state, global token count, per-rank example cursor, and immutable run configuration.

## Fault contract

The training process writes its own PID inside the pod. After initial progress, the harness reads that PID and sends `SIGSTOP` to rank 3, then verifies:

1. Linux reports the worker in stopped state `T`.
2. The same pod UID still exists and Kubernetes reports it `Running`.
3. Rank 0's latest step and durable checkpoint remain unchanged for 30 seconds.

The stable pair becomes the recovery fence. Only then does the harness inject synthetic DCGM field 230 value 79. Passing requires controller-owned isolation and eviction of that exact stopped pod; without the controller, the pod remains and the test times out. Every recreated rank must report the exact fence checkpoint, and recovered progress must exceed the stalled step.

## Observed timeline

| Time (PDT) | Elapsed | Observation |
| --- | ---: | --- |
| 19:09:20 | 0.0 s | Harness started the four-rank SmolLM recovery test. |
| 19:09:22 | 2.4 s | Kubernetes accepted `TrainJob` `aws-train-20260825020920`. |
| 19:09:35 | 15.7 s | Ranks 0–3 were placed on four distinct L4 nodes. |
| 19:10:39 | 79.1 s | All four ranks proved SmolLM3-3B, CUDA, NCCL, and `world_size=4`. |
| 19:11:27 | 126.9 s | Rank 0 reported step 11 and durable EFS checkpoint 10. |
| 19:11:35 | 135.7 s | The harness stopped rank 3's Python worker on `gpu-d`. |
| 19:12:14 | 174.6 s | The same pod still reported `Running`; step 15/checkpoint 15 had remained stable for 30 seconds. |
| 19:12:17 | 177.2 s | The harness injected synthetic XID 79 telemetry on `gpu-d`. |
| 19:12:21 | 181.3 s | The controller detected XID 79 and isolated `gpu-d`. |
| 19:12:25 | 185.6 s | Kubernetes observed eviction of the stopped rank 3 pod. |
| 19:12:26 | 186.6 s | Replacement rank 3 appeared on `gpu-e`. |
| 19:13:50 | 270.8 s | Every recreated rank resumed exactly checkpoint 15. |
| 19:13:54 | 274.6 s | Recovered training reached step 19. |
| 19:14:17 | 297.5 s | An explicit zero survived the confirmation window and returned `gpu-d` to service. |
| 19:14:23 | 303.2 s | Harness completed `PASS`. |

Detection took 4.1 seconds from telemetry injection to isolation, replacement took 5.3 seconds from injection, and workload recovery took 93.3 seconds from isolation to new progress. Warm image and model caches kept the recovery path well below the earlier cold-cache run.

Checkpoint 15 superseded checkpoint 10 while the harness completed pre-fault health checks, before rank 3 was paused. After confirming the pause, the harness observed the unchanged step/checkpoint pair for 30 seconds, then injected telemetry and required an exact checkpoint-15 restore.

## What the result supports

- A pod can remain Kubernetes `Running` while one GPU rank has stopped and the distributed workload cannot progress.
- Accelerator telemetry can close that observability gap and drive an explicit, opt-in remediation policy.
- The controller's taint-then-evict path, rather than ordinary pod failure, caused replacement of the stopped worker.
- Trainer/JobSet group recreation plus application-owned EFS checkpoints can restore consistent state on a different physical GPU.
- A stable recovery fence and exact-resume assertion distinguish real recovery from merely accepting any newer checkpoint.

## What it does not support

- Neither `SIGSTOP` nor DCGM field injection is a natural driver or hardware failure; the experiment does not validate the device's behavior during a real XID 79.
- The test does not measure scaling efficiency, convergence equivalence, model quality, or failure behavior inside a CUDA kernel.
- LoRA state is much smaller than full-parameter or sharded optimizer state, so this does not validate large-checkpoint throughput.
- JobSet recreates the group; this is not elastic membership or in-place rank repair.

The harness cleared the injected telemetry and deleted the `TrainJob`. Destroying the Terraform-managed AWS environment was a separate manual action outside this experiment.
