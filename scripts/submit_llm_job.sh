#!/bin/bash
#SBATCH --job-name=llm-lora-tune
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=compute
#SBATCH --output=/shared/logs/llm-lora-%j.out

set -e

mkdir -p /shared/logs /shared/models /shared/datasets /shared/ray_results

echo "==========================================="
echo "  LLM LoRA Fine-Tuning with Ray + Slurm"
echo "==========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Nodes:     $SLURM_JOB_NODELIST"
echo "Num Nodes: $SLURM_JOB_NUM_NODES"
echo "Model:     distilgpt2 + LoRA"
echo "==========================================="

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node" | awk '{print $1}')

echo "Head node: $head_node ($head_node_ip)"

echo "Starting Ray head on $head_node..."
ray start --head --port=6379 --num-cpus=2
sleep 5

for ((i = 1; i < ${#nodes_array[@]}; i++)); do
    worker=${nodes_array[$i]}
    echo "Starting Ray worker on $worker..."
    srun --nodes=1 --ntasks=1 -w "$worker" \
        ray start --address="$head_node_ip:6379" --num-cpus=2 &
done

sleep 10

echo ""
echo "Ray cluster status:"
ray status
echo ""

echo "Starting LoRA hyperparameter search..."
python3 /app/src/finetune_llm.py

echo ""
echo "==========================================="
echo "  Done! Results in /shared/ray_results/"
echo "==========================================="

ray stop
