import uuid

import torch
from model import SimpleNN
from seml import Experiment
from seml.database import get_mongodb_config
from seml.utils import ASHA  # Import asha class
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# def seed_everything(job_id):
#     # Combine job_id and entropy to make a unique seed
#     entropy = f"{job_id}-{time.time()}-{os.urandom(8)}".encode("utf-8")
#     seed = int(hashlib.sha256(entropy).hexdigest(), 16) % (2**32)
#     print(f"[Seed] Using seed: {seed}")
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     return seed

experiment = Experiment()


@experiment.config
def default_config():
    num_stages = 10
    dataset = "mnist"
    hidden_units = [64]
    dropout = 0.3
    learning_rate = 1e-3
    base_shared_dir = "./shared/experiments"  # Parent directory shared across jobs
    job_id = None  # Must be unique per job
    seed = 42
    asha_collection_name = "unknown_experiment"
    samples = 5
    asha = {"eta": 3, "min_r": 1, "max_resource": 20, "progression": "increase"}


@experiment.automain
def main(
    num_stages,
    dataset,
    hidden_units,
    dropout,
    learning_rate,
    base_shared_dir,
    job_id,
    asha,
    _log,
    _run,
):
    mongodb_configurations = get_mongodb_config()

    print(
        f"job parameters, hiddenunits:{hidden_units}, dropout:{dropout}, learningrate:{learning_rate}"
    )
    if job_id is None:
        job_id = str(uuid.uuid4())
        # job_id = str(_run._id)

    asha_collection_name = _run.config.get("asha_collection_name", "unknown_experiment")
    print("Run info:", _run.experiment_info)

    # Create model
    model = SimpleNN(hidden_units=hidden_units, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare dataset and loaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    full_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ASHA setup
    eta = asha["eta"]
    min_r = asha["min_r"]
    max_r = asha["max_resource"]
    metric_increases = asha["metric_increases"]
    tracker = ASHA(
        asha_collection_name=asha_collection_name,
        eta=eta,
        min_r=min_r,
        max_r=max_r,
        metric_increases=metric_increases,
        mongodb_configurations=mongodb_configurations,
        _log=_log,
    )
    for stage in range(num_stages):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        metric = correct / total
        print(f"[Epoch {stage}] Validation Accuracy: {metric:.4f}")

        if stage < (num_stages - 1):
            should_stop = tracker.store_stage_metric(stage, metric)
            if should_stop:
                print("We should end this process here")
                print(
                    f"job parameters, hiddenunits:{hidden_units}, dropout:{dropout}, learningrate:{learning_rate}"
                )
                break
        else:
            print("job finished")
            print(
                f"job parameters, hiddenunits:{hidden_units}, dropout:{dropout}, learningrate:{learning_rate}"
            )

    return {
        "asha_collection_name": asha_collection_name,
        "job_id": job_id,
        "metric_history": tracker.metric_history,
        "final_metric": tracker.metric_history[-1],
        "hidden_units": hidden_units,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "num_stages": num_stages,
        "dataset": dataset,
        "device": str(device),
        "final_stage": len(tracker.metric_history),
    }
