#!/bin/bash
set -e

echo "[controller] Generating munge key..."
dd if=/dev/urandom bs=1 count=1024 > /shared/munge.key 2>/dev/null
cp /shared/munge.key /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

echo "[controller] Starting munge..."
munged --force
sleep 2

echo "[controller] Starting slurmctld..."
slurmctld

echo "[controller] Ready. Waiting for workers..."
sleep 10

echo "[controller] Cluster status:"
sinfo
echo ""
scontrol show nodes

echo "==========================================="
echo "  Slurm controller is running."
echo "  Submit jobs with: sbatch /app/scripts/submit_ray_job.sh"
echo "==========================================="

tail -f /dev/null
