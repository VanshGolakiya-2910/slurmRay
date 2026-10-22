# Code Walkthrough — Where Slurm and Ray Are Used

A file-by-file guide to this repo, in the order things actually run: build the image → start the cluster → submit the job → run the Python.

Labels used throughout:

- **[SLURM]** — Slurm (the scheduler that hands out machines)
- **[RAY]** — Ray (the framework that spreads Python work across those machines)
- **[PLAIN]** — ordinary Docker / bash / PyTorch

## Map

| File | Slurm | Ray | Job |
|---|---|---|---|
| `docker/Dockerfile` | installs | installs | One image used by every node |
| `docker-compose.yml` | – | – | Creates the 3 "machines" |
| `config/slurm.conf` | ✅ all | – | Describes the cluster to Slurm |
| `scripts/start-controller.sh` | ✅ | – | Starts Slurm's brain |
| `scripts/start-worker.sh` | ✅ | – | Starts Slurm's per-node agent |
| `scripts/submit_ray_job.sh` | ✅ | ✅ | **Where they meet**: Slurm job → Ray cluster |
| `src/tune_mnist.py` | – | ✅ all | The AI work (hyperparameter search) |

The key idea: **Slurm never touches the Python, and the Python never mentions Slurm.** The only file that knows about both is `submit_ray_job.sh`.

```
sbatch ──► Slurm allocates worker1 + worker2
              │
              ▼
   submit_ray_job.sh runs on worker1
     ├─ ray start --head            (worker1 = Ray head)
     ├─ srun -w worker2 ray start   (worker2 joins)
     ├─ python3 tune_mnist.py       (Ray Tune spreads 12 trials over 4 CPUs)
     └─ ray stop                    (job ends, Slurm frees the nodes)
```

---

## 1. `docker/Dockerfile` — the one image

```dockerfile
FROM ubuntu:22.04
RUN apt-get install -y slurm-wlm slurm-client munge python3 python3-pip   # [SLURM] + munge
RUN mkdir -p /var/spool/slurmd /var/spool/slurmctld /var/log/slurm ...    # [SLURM] state & log dirs
RUN chown slurm:slurm ... && chown munge:munge ...                        # daemons refuse wrong ownership
RUN pip3 install torch torchvision --index-url .../cpu                   # [PLAIN] PyTorch, CPU build
RUN pip3 install "ray[tune]"                                              # [RAY] Ray core + Tune
COPY config/slurm.conf /etc/slurm/slurm.conf                              # [SLURM] same config on every node
COPY scripts/ /app/scripts/
```

- All three containers use this image, so every node has both Slurm and Ray installed. A node's role depends only on which startup script it runs.
- `slurm-wlm` provides `slurmctld` (controller daemon) and `slurmd` (node agent). `munge` is how Slurm nodes authenticate each other.
- `ray[tune]` installs core Ray plus Tune's extra dependencies.
- On a real cluster Slurm is installed on the machines once; Docker only fakes the machines.

## 2. `docker-compose.yml` — the fake cluster

```yaml
services:
  controller:
    hostname: controller                         # must match SlurmctldHost in slurm.conf
    command: /app/scripts/start-controller.sh    # role = controller
    volumes: [shared:/shared, ./src:/app/src]
  worker1:
    hostname: worker1                            # must match NodeName=worker1 in slurm.conf
    command: /app/scripts/start-worker.sh        # role = compute node
  worker2: ...                                   # same as worker1
volumes: { shared: }                             # one disk seen by all three
networks: { slurm-net: { driver: bridge } }      # private network; hostnames resolve to IPs
```

- **Hostnames matter.** Slurm finds nodes by name; `worker1` here must equal `NodeName=worker1` in `slurm.conf`, or the node never registers.
- **`shared` volume** = a real cluster's shared filesystem (NFS, Lustre). It holds the munge key, logs, the dataset and Ray results.
- **`./src:/app/src`** mounts the Python live, so you can edit `tune_mnist.py` without rebuilding.

## 3. `config/slurm.conf` — [SLURM]

```ini
ClusterName=demo-cluster
SlurmctldHost=controller          # which machine runs the brain (slurmctld)

SchedulerType=sched/backfill      # slot small jobs into gaps while a big one waits
SelectType=select/cons_tres       # hand out individual CPUs/memory/GPUs, not whole nodes
SelectTypeParameters=CR_Core      # ...counted by CPU core

AuthType=auth/munge               # every message authenticated by munge
AccountingStorageType=accounting_storage/none   # no history DB, so `sacct` won't work
ReturnToService=2                 # a DOWN node rejoins automatically once healthy

NodeName=worker1 CPUs=2 RealMemory=1024 State=UNKNOWN   # inventory: 2 nodes × 2 CPUs × 1 GB
NodeName=worker2 CPUs=2 RealMemory=1024 State=UNKNOWN

PartitionName=compute Nodes=worker1,worker2 Default=YES MaxTime=INFINITE State=UP
```

- **`NodeName` lines are Slurm's inventory.** Slurm only hands out what's declared. The job's `--cpus-per-task=2` works because each node declares `CPUs=2`; ask for 3 and the job stays PENDING forever.
- **A partition is a named queue** over a set of nodes. `#SBATCH --partition=compute` refers to it; `sinfo` shows it.
- Everything else (PID files, spool dirs, timeouts) is housekeeping you rarely change.
- For GPUs (Phase 2) this file gains `GresTypes=gpu` and `Gres=gpu:N` on each `NodeName` line.

## 4. `scripts/start-controller.sh` — [SLURM]

```bash
dd if=/dev/urandom bs=1 count=1024 > /shared/munge.key   # 1. random 1 KB secret...
cp /shared/munge.key /etc/munge/munge.key                #    ...where munge expects it
chown munge:munge ...; chmod 400 ...                     #    munge rejects readable keys
munged --force                                           # 2. start munge
slurmctld                                                # 3. start Slurm's brain
sleep 10; sinfo; scontrol show nodes                     # 4. print state to the container log
tail -f /dev/null                                        # 5. keep the container alive
```

- **Munge is a shared password.** Every node needs the same key or they reject each other. The controller writes it to `/shared` so workers can copy it.
- `slurmctld` daemonizes and returns, so `tail -f /dev/null` keeps the container from exiting.
- `sinfo` / `scontrol show nodes` only print status, visible with `docker logs controller`.

## 5. `scripts/start-worker.sh` — [SLURM]

```bash
while [ ! -f /shared/munge.key ]; do sleep 1; done   # wait for the controller's key
cp /shared/munge.key /etc/munge/munge.key            # use the SAME key
munged --force
slurmd                                                # node agent: reads slurm.conf, registers with controller
tail -f /dev/null
```

- A simple "wait for a file" handshake. Compose's `depends_on` only orders startup; it doesn't wait for the controller to be *ready*.
- Once `slurmd` registers, the node shows as `idle` in `sinfo`.

> At this point there is a plain Slurm cluster and **no Ray anywhere**. Ray is created per job, inside the next file.

---

## 6. `scripts/submit_ray_job.sh` — where Slurm meets Ray ⭐

Submitted with `sbatch`. It runs **on worker1**, not the controller: Slurm always runs a batch script on the first allocated node.

### A. [SLURM] Resource request

```bash
#!/bin/bash
#SBATCH --job-name=ray-tune-demo        # name shown in squeue
#SBATCH --nodes=2                       # 2 machines
#SBATCH --ntasks-per-node=1             # one task per machine
#SBATCH --cpus-per-task=2               # 2 CPUs each → 4 total
#SBATCH --partition=compute             # queue from slurm.conf
#SBATCH --output=/shared/logs/ray-tune-%j.out   # %j → job ID
```

`#SBATCH` lines are comments to bash but instructions to `sbatch`, read before the script runs. The job stays PENDING until Slurm can grant all of it.

### B. [SLURM] What did I get?

```bash
echo "Job ID:    $SLURM_JOB_ID"          # set by Slurm, e.g. 1
echo "Nodes:     $SLURM_JOB_NODELIST"    # compact form: worker[1-2]

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")        # → "worker1 worker2"
nodes_array=($nodes)                                           # bash array
head_node=${nodes_array[0]}                                    # first node → Ray head
head_node_ip=$(getent hosts "$head_node" | awk '{print $1}')   # hostname → IP
```

Slurm talks to your script through `SLURM_*` environment variables. The node list is compressed (`gpu[01-16]` on a big cluster); `scontrol show hostnames` is the standard way to expand it. No node names are hard-coded, so the script works on whatever nodes Slurm gives you.

### C. [RAY] Start the head here

```bash
ray start --head --port=6379 --num-cpus=2
sleep 5
```

We're already on worker1, so **worker1 becomes the Ray head**: GCS (cluster registry) on port 6379, dashboard, raylet. `--num-cpus=2` limits Ray to what Slurm allocated; without it Ray grabs every core on the machine, which on a shared cluster steals other jobs' CPUs.

### D. [SLURM + RAY] Start Ray on every other node ⭐

```bash
for ((i = 1; i < ${#nodes_array[@]}; i++)); do              # every node except index 0
    worker=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w "$worker" \                 # [SLURM] run a command ON that node
        ray start --address="$head_node_ip:6379" --num-cpus=2 &   # [RAY] join the head
done
sleep 10
```

- **`srun`** runs a command on another node *inside this job's allocation*. Unlike `ssh`, Slurm knows about the process and can account for and clean it up. `-w` picks the node.
- **`ray start --address=IP:6379`** joins that node to the head.
- **`&`** backgrounds it, so with 10 nodes all 9 workers start in parallel.
- **`sleep 10`** is a crude wait; a sturdier version loops on `ray status` until the expected node count appears.

### E. [RAY] Check, run, clean up

```bash
ray status                         # expect 2 nodes, 4.0 CPUs
python3 /app/src/tune_mnist.py     # the AI work
ray stop                           # stop Ray on this node
```

When the script exits the job is COMPLETED and both nodes return to `idle`. **One Slurm job = one temporary Ray cluster.**

> Note: `ray stop` runs only on worker1. Ray on worker2 was started by `ray start`, which daemonizes, so it may outlive the job unless Slurm kills it. If a later run has trouble on worker2, add `srun ray stop` across the nodes before the final `ray stop`. (Not verified against this demo.)

---

## 7. `src/tune_mnist.py` — [RAY]

No Slurm here at all; to Python, the cluster is just "Ray, somewhere." Two parts: `train_mnist(config)` trains **one** model; `main()` tells Ray Tune to run it many times with different configs.

### `main()` — the Ray side

```python
ray.init(address="auto")
```
**[RAY]** Connects to the already-running cluster (the head on this same node); it doesn't start a new one. `ray.init()` with no address starts a one-machine cluster — that's how you'd test on a laptop.

```python
datasets.MNIST("/shared/data", train=True, download=True)
datasets.MNIST("/shared/data", train=False, download=True)
```
**[PLAIN]** Download once to the shared disk up front, so 4 trials starting together don't all download and collide.

```python
scheduler = ASHAScheduler(
    metric="loss", mode="min",   # judge by loss; lower is better
    max_t=10,                    # no trial exceeds 10 reports (= 10 epochs)
    grace_period=2,              # everyone gets at least 2 epochs
    reduction_factor=2,          # at each checkpoint keep roughly the better half
)
```
**[RAY]** The early-stopping rule. Trials are judged at epochs 2, 4, 8; worse-half trials stop and free their CPU for a new trial.

```python
search_space = {
    "lr":          tune.loguniform(1e-4, 1e-1),       # random, even on a log scale
    "momentum":    tune.uniform(0.1, 0.9),            # random, even
    "batch_size":  tune.choice([32, 64, 128, 256]),   # pick one
    "hidden_size": tune.choice([64, 128, 256, 512]),
}
```
**[RAY]** Describes *how to draw* values; each trial gets its own random combination. Log-uniform suits learning rates because 0.001→0.01 matters as much as 0.01→0.1.

```python
result = tune.run(
    train_mnist,                      # the function to run many times
    config=search_space,
    num_samples=12,                   # 12 trials
    scheduler=scheduler,
    resources_per_trial={"cpu": 1},   # 1 CPU each → 4 CPUs = 4 at once
    storage_path="/shared/ray_results",
    name="mnist_tune_demo",
    raise_on_failed_trial=False,      # one crash doesn't kill the search
)
```
**[RAY]** This is where the distribution happens. Tune turns each trial into a Ray actor and Ray places it wherever a CPU is free — worker1 or worker2. You never pick nodes. When a trial ends or is stopped, the freed CPU takes the next one.

```python
best = result.get_best_trial("loss", "min", "last")   # lowest loss by each trial's last report
```

### `train_mnist(config)` — plain PyTorch plus one Ray line

```python
def train_mnist(config):                         # config = ONE draw, e.g. {"lr": 0.003, ...}
    train_loader = DataLoader(train_data, batch_size=int(config["batch_size"]))
    model = MNISTNet(int(config["hidden_size"]))
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()                    # clear old gradients
            loss = criterion(model(data), target)    # forward pass + loss
            loss.backward()                          # gradients
            optimizer.step()                         # update weights

        model.eval()                                 # score on the test set
        ...
        tune.report({"loss": ..., "accuracy": correct / total})   # [RAY]
```

- Apart from `tune.report`, it's an ordinary PyTorch loop; you could call `train_mnist({...})` directly without Ray.
- **`tune.report` is the two-way link**: it sends this epoch's score to Tune (that's how ASHA sees progress), and if ASHA decides the trial is losing, the trial stops there. 10 reports is why `max_t=10` means 10 epochs.
- **`MNISTNet`**: flatten 28×28 → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear(10). `hidden_size` sets the middle-layer width.

---

## 8. `llm-lora` branch — same pattern, different model

Check it out with `git checkout llm-lora`.

**`scripts/submit_llm_job.sh`** is a near copy of `submit_ray_job.sh`; only the job name, log name and Python file differ. The Slurm and Ray plumbing is unchanged, which is the point: the pattern is reusable.

**`src/finetune_llm.py`** mirrors `tune_mnist.py`:

| | `tune_mnist.py` | `finetune_llm.py` |
|---|---|---|
| `ray.init(address="auto")` | ✅ | ✅ |
| Pre-download to `/shared` | MNIST | distilgpt2 + wikitext-2 |
| Search space | lr, momentum, batch, hidden | LoRA rank `r`, `lora_alpha`, lr, dropout, batch |
| ASHA | grace 2, max 10 | grace 1, max 3 (epochs are slow) |
| Trials / per trial | 12 / `{"cpu": 1}` | 8 / `{"cpu": 2}` → 2 at once |
| `tune.report` | loss, accuracy | loss, perplexity |

The new non-Ray part is LoRA:

```python
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
lora_config = LoraConfig(
    r=config["lora_r"],                  # adapter rank: bigger = more capacity
    lora_alpha=config["lora_alpha"],     # how strongly the adapter affects output
    target_modules=["c_attn", "c_proj"], # GPT-2's attention projections get adapters
    lora_dropout=config["lora_dropout"],
    task_type="CAUSAL_LM",               # next-token prediction
)
model = get_peft_model(model, lora_config)   # freeze original weights, add trainable adapters
```

Only the small adapters train (the script prints the trainable %, about 0.49%). Training uses `labels = input_ids`, so the model learns to predict the next token; `perplexity = exp(loss)` measures how well — lower is better.

---

## Check yourself

1. **Which machine runs `submit_ray_job.sh`?** — worker1 (the first allocated node), not the controller.
2. **What breaks if you remove `--num-cpus=2` from both `ray start` lines?** — Probably nothing visible here; on a real shared cluster Ray would claim CPUs Slurm gave to other jobs.
3. **How would you run 8 trials at once?** — More CPUs/nodes in `slurm.conf`, request them via `#SBATCH` and `--num-cpus`, or lower `resources_per_trial`; raise `num_samples` so there are enough trials.

**Exercise:** change `num_samples=12` to `4` in `src/tune_mnist.py`, rerun, and compare `/shared/logs/ray-tune-<id>.out` with the first run. No rebuild needed — `src/` is mounted live.
