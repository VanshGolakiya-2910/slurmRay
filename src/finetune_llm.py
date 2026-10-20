"""
Distributed LoRA Hyperparameter Search for LLM Fine-Tuning.
Uses Ray Tune + Slurm to search over LoRA configs on distilgpt2.
"""

import math
import torch
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL_NAME = "distilgpt2"
CACHE_DIR = "/shared/models"
DATA_DIR = "/shared/datasets"
MAX_LENGTH = 128
NUM_TRAIN_SAMPLES = 300
NUM_VAL_SAMPLES = 80
NUM_EPOCHS = 3


def train_llm_lora(config):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    lora_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        target_modules=["c_attn", "c_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=DATA_DIR, trust_remote_code=True)

    def tokenize_and_filter(split, n_samples):
        subset = split.select(range(min(n_samples * 3, len(split))))
        tokenized = subset.map(
            lambda ex: tokenizer(
                ex["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length",
            ),
            batched=True,
            remove_columns=["text"],
        )
        tokenized = tokenized.filter(
            lambda x: sum(1 for t in x["input_ids"] if t != tokenizer.eos_token_id) > 10
        )
        tokenized = tokenized.select(range(min(n_samples, len(tokenized))))
        tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})
        tokenized.set_format("torch")
        return tokenized

    train_data = tokenize_and_filter(dataset["train"], NUM_TRAIN_SAMPLES)
    val_data = tokenize_and_filter(dataset["validation"], NUM_VAL_SAMPLES)

    train_loader = DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                total_loss += outputs.loss.item()
                count += 1

        avg_loss = total_loss / max(count, 1)
        perplexity = math.exp(min(avg_loss, 20))
        tune.report({"loss": avg_loss, "perplexity": perplexity})


def main():
    ray.init(address="auto")

    print("Pre-downloading model and dataset...")
    AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=DATA_DIR, trust_remote_code=True)
    print("Downloads complete.")

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=NUM_EPOCHS,
        grace_period=1,
        reduction_factor=2,
    )

    search_space = {
        "lora_r": tune.choice([4, 8, 16]),
        "lora_alpha": tune.choice([8, 16, 32]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "lora_dropout": tune.choice([0.0, 0.05, 0.1]),
        "batch_size": tune.choice([4, 8]),
    }

    result = tune.run(
        train_llm_lora,
        config=search_space,
        num_samples=8,
        scheduler=scheduler,
        resources_per_trial={"cpu": 2},
        storage_path="/shared/ray_results",
        name="llm_lora_tune",
        verbose=1,
        raise_on_failed_trial=False,
    )

    best = result.get_best_trial("loss", "min", "last")
    print("\n===========================================")
    print("  Best LoRA Fine-Tuning Trial")
    print("===========================================")
    print(f"  LoRA rank:    {best.config['lora_r']}")
    print(f"  LoRA alpha:   {best.config['lora_alpha']}")
    print(f"  Learning rate:{best.config['lr']:.6f}")
    print(f"  Dropout:      {best.config['lora_dropout']}")
    print(f"  Batch size:   {best.config['batch_size']}")
    print(f"  Loss:         {best.last_result['loss']:.4f}")
    print(f"  Perplexity:   {best.last_result['perplexity']:.2f}")
    print("===========================================")


if __name__ == "__main__":
    main()
