# Project Learnings

This is a running record of the mental models we build while developing the distributed training and GPU node health system.

## DistributedDataParallel (DDP)

- A **rank** is the unique ID of a training process. With two processes, the ranks are `0` and `1`, and the world size is `2`.
- Every DDP rank holds a complete copy of the model and runs every training step. Rank 0 conventionally handles logging and checkpoint writes so the ranks do not duplicate or race on that work.
- DDP divides the **data-side work**. Each rank runs the full model on a different local batch of samples. It does not split an individual matrix multiplication; that is tensor parallelism.
- During each step, every rank performs a forward pass and backward pass. DDP averages the gradients from the ranks, and then every rank applies the same optimizer update. This keeps all model replicas synchronized.
- The local fixture's optimizer is mini-batch stochastic gradient descent (`SGD`). In the geometric mental model, the parameters are a point in a many-dimensional space. Each rank estimates the downhill direction from different examples; averaging their gradients gives a direction based on the combined batch.
- In the local fixture, a batch of 16 examples per rank and two ranks means one synchronized step uses an effective global batch of 32 examples. Both ranks execute the step at the same time; they do not divide the step numbers between themselves.

```
DDP step 3:
  rank 0: 16 examples ─┐
                       ├ simultaneously → approximately ε + communication
  rank 1: 16 examples ─┘

DDP step 4:
  rank 0: 16 examples ─┐
                       ├ simultaneously → approximately ε + communication
  rank 1: 16 examples ─┘

Total: 64 examples in approximately 2ε + communication overhead
```

## Python versus `torchrun`

- `python train.py` launches one process. Without distributed environment variables, our script defaults to rank 0 and a world size of 1, so DDP is disabled.
- `torchrun --nproc-per-node=2 train.py` launches two copies of the script and gives each its rank, world size, and rendezvous settings. The script still has to initialize the process group and wrap the model in DDP.
- Our local smoke test uses `torchrun` to create two ranks in one container. In Kubernetes, Trainer v2 creates an indexed two-pod Job and supplies `PET_*` settings for node count, node rank, and rendezvous. Each pod expands those settings into a one-process `torchrun` invocation, which then supplies the standard `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` variables to the training script.
- After a force deletion, a replacement TCPStore client can remain connected to a dying rendezvous generation longer than rank 0's process-group timeout. The training process has an outer rendezvous watchdog so a stuck client exits and JobSet can retry the complete rank group into the same generation.

## Kubeflow Trainer v2 execution model

- A `TrainJob` is the user-facing request: image, command, node count, processes per node, and resources. It does not contain a full pod template.
- A `TrainingRuntime` is the platform-owned execution policy. This demo keeps it namespaced beside the TrainJob and uses it to define the JobSet template, checkpoint volume, placement preference, image pull behavior, and restart budget.
- The Torch policy converts `numNodes: 2` and `numProcPerNode: 1` into one indexed Kubernetes Job with completion indexes 0 and 1. JobSet labels identify the TrainJob, while `batch.kubernetes.io/job-completion-index` identifies the distributed rank pod.
- `backoffLimit: 0` makes a failed rank fail its child Job immediately; JobSet's `Recreate` strategy then restarts the distributed group together. `maxRestarts: 20` bounds that recovery loop.

## Optimizer Steps and Checkpoints

- Training advances through optimizer steps. A checkpoint is only a periodic snapshot of the synchronized model and optimizer state.
- In the local fixture, with `--checkpoint-every=10`, both ranks complete and synchronize ten optimizer steps between checkpoints. Rank 0 then records the snapshot.
- If training fails after step 17 and the latest checkpoint is step 10, the restarted ranks resume at step 10 and repeat steps 11 through 17.
- Test step numbers are arbitrary stopping points.

## Choosing a Parallelism Strategy

- DDP can run across machines, but the complete model must fit on every GPU.
- Tensor parallelism splits large tensor operations and their weights across devices.
- Pipeline parallelism places different groups of layers on different devices.
- FSDP or ZeRO shards parameters, gradients, and optimizer state. These are common choices when the model does not fit on one GPU.
