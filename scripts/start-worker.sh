#!/bin/bash
set -e

echo "[$(hostname)] Waiting for munge key..."
while [ ! -f /shared/munge.key ]; do
    sleep 1
done
sleep 2

cp /shared/munge.key /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

echo "[$(hostname)] Starting munge..."
munged --force
sleep 2

echo "[$(hostname)] Starting slurmd..."
slurmd

echo "[$(hostname)] Worker ready."
tail -f /dev/null
