#!/bin/bash
#SBATCH --job-name=ray-tune-demo
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=compute
#SBATCH --output=/shared/logs/ray-tune-%j.out

set -e

mkdir -p /shared/logs /shared/data /shared/ray_results

echo "==========================================="
echo "  Ray + Slurm Distributed Tuning Demo"
echo "==========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Nodes:     $SLURM_JOB_NODELIST"
echo "Num Nodes: $SLURM_JOB_NUM_NODES"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node" | awk '{print $1}')

echo "Head node: $head_node ($head_node_ip)"
echo "==========================================="

echo "Starting Ray head on $head_node..."
ray start --head --port=6379 --num-cpus=2 --dashboard-host=0.0.0.0
sleep 5

for ((i = 1; i < ${#nodes_array[@]}; i++)); do
    worker=${nodes_array[$i]}
    worker_ip=$(getent hosts "$worker" | awk '{print $1}')
    echo "Starting Ray worker on $worker ($worker_ip)..."
    srun --nodes=1 --ntasks=1 -w "$worker" \
        ray start --address="$head_node_ip:6379" --num-cpus=2 &
done

sleep 10

echo ""
echo "Ray cluster status:"
ray status
echo ""

echo "Starting hyperparameter tuning..."
python3 /app/src/tune_mnist.py

echo ""
echo "==========================================="
echo "  Done! Results in /shared/ray_results/"
echo "==========================================="

ray stop
