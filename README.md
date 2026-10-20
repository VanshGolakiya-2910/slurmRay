# SlurmRay — LLM LoRA Fine-Tuning Branch

Distributed LLM fine-tuning with **LoRA** on a Dockerized **Slurm** cluster using **Ray Tune** for hyperparameter search.

This branch extends the base demo ([`main`](https://github.com/VanshGolakiya-2910/slurmRay/tree/main)) with LLM fine-tuning capabilities.

## What This Does

- Fine-tunes **distilgpt2** (82M params) on **wikitext-2** using LoRA (only 0.49% params trainable)
- **Ray Tune** searches over LoRA hyperparameters (rank, alpha, learning rate, dropout) across the cluster
- **ASHA scheduler** early-stops bad configs to save compute
- **Slurm** manages resource allocation across Docker-based compute nodes

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
                    │   Ray Tune (8 trials)
                    │   LoRA configs distributed
                    │   across 4 CPUs
                    └──────────────────────────
```

## Search Space

| Parameter | Values |
|-----------|--------|
| LoRA rank | 4, 8, 16 |
| LoRA alpha | 8, 16, 32 |
| Learning rate | 1e-5 to 1e-3 (log uniform) |
| LoRA dropout | 0.0, 0.05, 0.1 |
| Batch size | 4, 8 |

## Quick Start

### Prerequisites

- Docker and Docker Compose

### Run

```bash
# Build images (includes transformers, peft, datasets)
docker compose build

# Start the 3-node Slurm cluster
docker compose up -d

# Verify nodes are ready
docker exec controller sinfo

# Submit the LoRA fine-tuning job
docker exec controller sbatch /app/scripts/submit_llm_job.sh

# Monitor progress
docker exec controller squeue                              # job status
docker exec controller tail -f /shared/logs/llm-lora-1.out # live output

# Cleanup
docker compose down -v
```

### Expected Output

- 8 trials run across 2 worker nodes (2 parallel at a time)
- ASHA early-stops ~half the trials after 1 epoch
- Best trials run all 3 epochs
- Total time: ~25 minutes on CPU
- Best perplexity: ~20 (varies by run)

## Project Structure

```
├── docker-compose.yml          # 3-node cluster (1 controller + 2 workers)
├── docker/Dockerfile           # Slurm + Ray + PyTorch + Transformers + PEFT
├── config/slurm.conf           # Slurm cluster configuration
├── scripts/
│   ├── start-controller.sh     # Controller entrypoint
│   ├── start-worker.sh         # Worker entrypoint
│   ├── submit_ray_job.sh       # MNIST hyperparameter tuning (from main)
│   └── submit_llm_job.sh       # LLM LoRA fine-tuning job
└── src/
    ├── tune_mnist.py            # MNIST tuning script (from main)
    └── finetune_llm.py          # LoRA fine-tuning with Ray Tune
```

## Key Concepts Demonstrated

- **Slurm** — Cluster resource manager scheduling jobs across nodes
- **Ray Tune** — Distributed hyperparameter search framework
- **LoRA** — Parameter-efficient fine-tuning (trains <1% of model weights)
- **ASHA** — Asynchronous successive halving for early stopping bad trials
- **Docker** — Portable cluster that runs on any machine

## See Also

- [`main` branch](https://github.com/VanshGolakiya-2910/slurmRay/tree/main) — Simpler MNIST hyperparameter tuning demo
