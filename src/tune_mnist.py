"""
Distributed Hyperparameter Tuning with Ray Tune on MNIST.
Submitted as a Slurm job across Docker-based compute nodes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


class MNISTNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x):
        return self.net(self.flatten(x))


def train_mnist(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_data = datasets.MNIST("/shared/data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("/shared/data", train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = MNISTNet(int(config["hidden_size"]))
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                correct += output.argmax(1).eq(target).sum().item()
                total += target.size(0)

        tune.report({"loss": test_loss / len(test_loader), "accuracy": correct / total})


def main():
    ray.init(address="auto")

    datasets.MNIST("/shared/data", train=True, download=True)
    datasets.MNIST("/shared/data", train=False, download=True)

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=2,
        reduction_factor=2,
    )

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "hidden_size": tune.choice([64, 128, 256, 512]),
    }

    result = tune.run(
        train_mnist,
        config=search_space,
        num_samples=12,
        scheduler=scheduler,
        resources_per_trial={"cpu": 1},
        storage_path="/shared/ray_results",
        name="mnist_tune_demo",
        verbose=1,
        raise_on_failed_trial=False,
    )

    best = result.get_best_trial("loss", "min", "last")
    print("\n===========================================")
    print("  Best Trial Results")
    print("===========================================")
    print(f"  Config:   {best.config}")
    print(f"  Loss:     {best.last_result['loss']:.4f}")
    print(f"  Accuracy: {best.last_result['accuracy']:.4f}")
    print("===========================================")


if __name__ == "__main__":
    main()
