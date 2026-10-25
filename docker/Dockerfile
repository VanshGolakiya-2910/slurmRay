FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    slurm-wlm \
    slurm-client \
    munge \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
    /var/spool/slurmd \
    /var/spool/slurmctld \
    /var/log/slurm \
    /var/run/munge \
    /etc/munge \
    /shared \
    /app/scripts \
    /app/src

RUN chown slurm:slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm \
    && chown munge:munge /var/run/munge /etc/munge

RUN pip3 install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir "ray[default,tune]"

COPY config/slurm.conf /etc/slurm/slurm.conf
RUN ln -sf /etc/slurm /etc/slurm-llnl 2>/dev/null || true

COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

VOLUME /shared
