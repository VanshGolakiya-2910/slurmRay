# SlurmRay — Distributed Hyperparameter Tuning Demo

A demo of **Slurm + Ray Tune** running distributed hyperparameter search on a Dockerized cluster.

Slurm manages compute nodes, Ray distributes the workload, and Docker makes the whole thing portable — no real cluster needed.

## What This Does

- Spins up a **3-node Slurm cluster** in Docker (1 controller + 2 workers)
- Submits a Slurm job that launches a **Ray cluster** across both workers
- Runs **Ray Tune** with ASHA scheduler to search hyperparameters for a CNN on MNIST
- Early-stops bad configs, runs 12 trials across 4 CPUs, finds best accuracy in ~4 minutes

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Controller  │     │   Worker 1  │     │   Worker 2  │
│  (slurmctld) │────▶│  (slurmd)   │     │  (slurmd)   │
│              │     │  Ray Head   │◀───▶│  Ray Worker  │
│  sbatch jobs │     │  2 CPUs     │     │  2 CPUs      │
└─────────────┘     └─────────────┘     └─────────────┘
                          │                    │
                    ┌─────┴────────────────────┘
                    │   Ray Tune (12 trials)
                    │   4 parallel, ASHA early-stopping
                    └──────────────────────────
```

## Quick Start

### Prerequisites

- Docker and Docker Compose

### Run

```bash
# Build images
docker compose build

# Start the Slurm cluster
docker compose up -d

# Verify both workers are ready
docker exec controller sinfo

# Submit the hyperparameter tuning job
docker exec controller sbatch /app/scripts/submit_ray_job.sh

# Monitor
docker exec controller squeue                              # job status
docker exec controller tail -f /shared/logs/ray-tune-1.out # live output

# Cleanup
docker compose down -v
```

### Expected Output

- 12 trials with random hyperparameter configs
- 4 trials run in parallel (1 per CPU)
- ASHA early-stops bad trials after 2 epochs, good ones run all 10
- Best accuracy: ~98% on MNIST
- Total time: ~4 minutes

## Search Space

| Parameter | Values |
|-----------|--------|
| Learning rate | 1e-4 to 1e-1 (log uniform) |
| Momentum | 0.1 to 0.9 (uniform) |
| Batch size | 32, 64, 128, 256 |
| Hidden size | 64, 128, 256, 512 |

## Project Structure

```
├── docker-compose.yml          # 3-node cluster (1 controller + 2 workers)
├── docker/Dockerfile           # Slurm + Ray + PyTorch image
├── config/slurm.conf           # Slurm cluster configuration
├── scripts/
│   ├── start-controller.sh     # Controller entrypoint
│   ├── start-worker.sh         # Worker entrypoint
│   └── submit_ray_job.sh       # Slurm batch job script
└── src/
    └── tune_mnist.py            # Ray Tune hyperparameter search on MNIST
```

## Key Concepts Demonstrated

- **Slurm** — Cluster resource manager scheduling jobs across nodes
- **Ray Tune** — Distributed hyperparameter search framework
- **ASHA Scheduler** — Asynchronous successive halving for early stopping bad trials
- **Docker** — Portable multi-node cluster on a single machine

## Branches

| Branch | Description |
|--------|-------------|
| [`main`](https://github.com/VanshGolakiya-2910/slurmRay/tree/main) | MNIST hyperparameter tuning (this branch) |
| [`llm-lora`](https://github.com/VanshGolakiya-2910/slurmRay/tree/llm-lora) | LLM fine-tuning with LoRA on distilgpt2 |
