"""CIFAR-10 + ResNet-18: the entry point that ties the whole project together.

Pick what to run with stage=...:
    stage=data   (Week 1) load + preprocess the data, run sanity checks, save augmentation samples
    stage=model  (Week 2) inspect the network and check that it can overfit a single batch
    stage=train  (Week 2) train the network and watch the training loss/accuracy
    stage=full   (Week 3) train with validation, keep the best checkpoint, evaluate on the test set

Edit CONFIG below, or override any key from the command line:
    python main.py stage=data augment=strong run_name=w1_strong
    python main.py stage=full epochs=15 lr=0.05 dropout=0.3 run_name=w3_dropout

This file is handed out in Week 1 but is completed over three weeks.
Every TODO is tagged with the week it belongs to:
    TODO [Week 1]  -> do now (needed for stage=data)
    TODO [Week 2]  -> leave for next week (needs resnet.py)
    TODO [Week 3]  -> leave for the last week (needs evaluation.py)
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data

# resnet (Week 2) and evaluation (Week 3) are imported inside the functions that need them,
# so that `stage=data` already works in Week 1 when those files do not exist yet.

CONFIG = {
    "stage": "full",           # "data", "model", "train", "full"
    # Data / preprocessing
    "data_dir": "./data",
    "val_split": 0.1,          # fraction of training data held out for validation
    "subset_fraction": 1.0,    # < 1.0 trains on less data (faster, shows data-size effect)
    "normalize": True,         # per-channel mean/std normalization
    "augment": "crop_flip",    # "none", "flip", "crop_flip", "strong"
    "label_smoothing": 0.0,    # 0.0 = hard one-hot; try 0.1
    "leak": "none",            # "none", "val_in_train", "test_as_val"  (Week 3 leakage demos)
    "batch_size": 128,
    "num_workers": 2,
    # Model
    "stem": "cifar",           # "cifar" (3x3, no maxpool) or "imagenet" (7x7 + maxpool)
    "width": 64,               # 64 = standard ResNet-18; try 32 or 16 for a smaller/faster model
    "residual": True,          # False removes all skip connections ("plain" network)
    "dropout": 0.0,
    # Training
    "epochs": 30,
    "optimizer": "sgd",        # "sgd", "adam", "adamw"
    "lr": 0.1,                 # good start: sgd 0.1, adam/adamw 1e-3
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "cosine",     # "cosine", "step", "linear", "none"
    "overfit_steps": 100,      # stage=model: steps of the single-batch overfitting test
    "seed": 42,
    # Output
    "run_name": "baseline",
    "output_dir": "./runs",
}

STAGES = ("data", "model", "train", "full")
LEAKS = ("none", "val_in_train", "test_as_val")


def parse_overrides(config, args):
    """Apply command-line arguments like ["lr=0.01", "augment=none"] to config.

    Rules:
      - every arg must look like key=value, otherwise raise ValueError
      - the key must already exist in config, otherwise raise KeyError (catches typos!)
      - the value is converted to the type of the default value:
            int   "30"   -> 30        float "0.01" -> 0.01      str "adam" -> "adam"
            bool  "true"/"1"/"yes" -> True, anything else -> False
    """
    # TODO [Week 1] 1: loop over args, split each on the FIRST "=" (arg.split("=", 1)),
    #   validate, convert and store the value.
    #   Careful: bool must be checked BEFORE int, because isinstance(True, int) is True
    #   in Python, and bool("False") is True!
    raise NotImplementedError("TODO [Week 1] 1: parse_overrides")


def set_seed(seed):
    """Make runs reproducible: seed Python's `random`, NumPy and PyTorch."""
    # TODO [Week 1] 2: three lines - random.seed, np.random.seed, torch.manual_seed
    raise NotImplementedError("TODO [Week 1] 2: set_seed")


def get_device():
    """Return the fastest available device: CUDA GPU, then Apple-silicon GPU (MPS), then CPU."""
    # TODO [Week 1] 3: use torch.cuda.is_available() and torch.backends.mps.is_available()
    #   and return torch.device("cuda" / "mps" / "cpu").
    raise NotImplementedError("TODO [Week 1] 3: get_device")


def build_data(cfg, device):
    """Returns ((train_set, val_set, test_set), (train_loader, val_loader, test_loader))."""
    # TODO [Week 1] 4:
    #   a) call data.load_cifar10(...) passing the matching cfg values
    #      (data_dir, val_split, subset_fraction, normalize, augment, label_smoothing, seed, leak)
    #   b) call data.make_loaders(...) with cfg["batch_size"], cfg["num_workers"] and
    #      pin_memory=(device.type == "cuda")   (pinned memory only speeds up CUDA transfers)
    #   c) return both tuples
    raise NotImplementedError("TODO [Week 1] 4: build_data")


def run_data_stage(cfg, device, run_dir):
    """Week 1 output: dataset summary, loader speed and an image of augmented samples."""
    # TODO [Week 1] 5:
    #   a) (train_set, val_set, test_set), (train_loader, _, _) = build_data(cfg, device)
    #   b) data.describe(train_set, val_set, test_set)
    #   c) rate = data.benchmark_loader(train_loader); print it together with
    #      cfg["batch_size"] and cfg["num_workers"], e.g.
    #        "Loader speed: 16,444 images/s (batch_size=128, num_workers=2)"
    #   d) data.save_augmentation_grid(cfg["data_dir"], cfg["augment"], run_dir / "augmentations.png")
    #      and print where the image was saved.
    raise NotImplementedError("TODO [Week 1] 5: run_data_stage")


def build_model(cfg, device):
    import resnet

    # TODO [Week 2] 6:
    #   a) model = resnet.ResNet18(data.NUM_CLASSES, <stem>, <width>, <dropout>, <residual>)
    #      taking the values from cfg, and move it to `device` with .to(device)
    #   b) print a summary line, e.g.
    #        "ResNet-18 (residual, stem=cifar, width=64): 11,173,962 parameters"
    #      ("plain" instead of "residual" when cfg["residual"] is False;
    #       use resnet.count_parameters(model))
    #   c) return model
    raise NotImplementedError("TODO [Week 2] 6: build_model")


def run_model_stage(cfg, device):
    """Week 2 output: layer shapes, initial loss, gradient norms and a single-batch overfit test."""
    import resnet

    # TODO [Week 2] 7:
    #   a) model = build_model(cfg, device); resnet.trace_shapes(model)
    #   b) take ONE batch from the training loader of build_data(cfg, device)
    #      (hint: next(iter(train_loader))) and move images and targets to `device`
    #   c) under torch.no_grad(): compute logits = model(images) and compare
    #        resnet.soft_cross_entropy(logits, targets)   (yours)
    #        F.cross_entropy(logits, targets)            (PyTorch, accepts probability targets)
    #        np.log(data.NUM_CLASSES)                    (theory: loss of a uniform guess)
    #      print all three - the first two must match exactly.
    #   d) print every (name, norm) of resnet.gradient_norms(model, images, targets)
    #   e) optimizer = resnet.build_optimizer(model, cfg["optimizer"], cfg["lr"], cfg["momentum"], 0.0)
    #      losses = resnet.overfit_single_batch(model, images, targets, optimizer, cfg["overfit_steps"])
    #      print the first and the last loss.
    raise NotImplementedError("TODO [Week 2] 7: run_model_stage")


def train(cfg, model, train_loader, device, run_dir, monitor_loader=None, evaluate_fn=None):
    """Train for cfg["epochs"] epochs. With evaluate_fn, also validate every epoch and save
    the checkpoint with the best validation accuracy. Returns (history, best_checkpoint_path)."""
    import resnet

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time")}
    best_acc, best_path = -1.0, run_dir / "best_model.pt"

    # TODO [Week 2] 8a: build the optimizer and the scheduler with resnet.build_optimizer(...)
    #   and resnet.build_scheduler(...) using the cfg values.

    # TODO [Week 2] 8b: the epoch loop: for epoch in range(1, cfg["epochs"] + 1):
    #   1. start = time.time()
    #   2. record the current learning rate: history["lr"].append(optimizer.param_groups[0]["lr"])
    #   3. train_loss, train_acc = resnet.train_one_epoch(model, train_loader, optimizer, device)
    #   4. if scheduler is not None: scheduler.step()      (ONCE per epoch, AFTER training)
    #   5. append train_loss / train_acc to history
    #   6. build a progress line, e.g.
    #        "Epoch   3/30 | lr 0.0976 | train loss 1.2345 acc 0.5432"
    #   7. if evaluate_fn is not None:   -> TODO [Week 3] 9 (leave it for Week 3)
    #   8. history["epoch_time"].append(time.time() - start) and print the line + seconds

    # TODO [Week 3] 9: (inside the epoch loop, step 7) validation + checkpointing
    #   a) val_loss, val_acc = evaluate_fn(model, monitor_loader, device); append both to history
    #   b) add f" | val loss {val_loss:.4f} acc {val_acc:.4f}" to the progress line
    #   c) if val_acc > best_acc: update best_acc, save the checkpoint
    #        torch.save({"model_state": model.state_dict(), "epoch": epoch,
    #                    "val_acc": val_acc, "config": cfg}, best_path)
    #      and add "  * saved" to the progress line.

    # TODO [Week 2] 8c: after the loop, save the history as JSON:
    #   (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    raise NotImplementedError("TODO [Week 2] 8: train")
    return history, best_path


def run_train_stage(cfg, device, run_dir):
    """Week 2 output: training loss/accuracy per epoch (no validation yet)."""
    # TODO [Week 2] 10: get the train loader from build_data, build the model, call
    #   train(cfg, model, train_loader, device, run_dir)  (no evaluate_fn -> no validation)
    #   and print where history.json was saved.
    raise NotImplementedError("TODO [Week 2] 10: run_train_stage")


def run_full_stage(cfg, device, run_dir):
    """Week 3 output: training curves, best checkpoint, test accuracy, report, confusion matrix."""
    import evaluation

    _, (train_loader, val_loader, test_loader) = build_data(cfg, device)

    # TODO [Week 3] 11: choose `monitor_loader`, the data used to pick the best epoch:
    #   - cfg["leak"] == "test_as_val": print a WARNING and use test_loader (leakage demo!)
    #   - val_loader is None: raise ValueError (we need a validation set)
    #   - otherwise: val_loader

    # TODO [Week 3] 12: build the model and train it WITH validation:
    #   history, best_path = train(cfg, model, train_loader, device, run_dir,
    #                              monitor_loader, evaluation.evaluate)
    #   then evaluation.plot_history(history, run_dir / "training_curves.png")

    # TODO [Week 3] 13: restore the BEST checkpoint (not the last epoch!) and test it ONCE:
    #   a) checkpoint = torch.load(best_path, map_location=device)
    #   b) model.load_state_dict(checkpoint["model_state"])
    #   c) print which epoch it came from and its val acc
    #   d) test_loss, test_acc = evaluation.evaluate(model, test_loader, device); print them

    # TODO [Week 3] 14: detailed analysis of the test predictions:
    #   a) y_true, y_pred = evaluation.predict(model, test_loader, device)
    #   b) evaluation.report(..., data.CLASSES, run_dir / "classification_report.txt")
    #   c) evaluation.plot_confusion_matrix(..., data.CLASSES, run_dir / "confusion_matrix.png")
    #   d) results = evaluation.summarize(history, test_loss, test_acc) and save it as
    #      run_dir / "results.json" (json.dumps(results, indent=2))
    raise NotImplementedError("TODO [Week 3] 11-14: run_full_stage")


def main():
    # TODO [Week 1] 6:
    #   a) cfg = parse_overrides(dict(CONFIG), sys.argv[1:])
    #      (dict(CONFIG) makes a copy, so the defaults above are never modified)
    #   b) raise ValueError if cfg["stage"] is not in STAGES or cfg["leak"] is not in LEAKS
    #   c) set_seed(cfg["seed"]); device = get_device()
    #   d) run_dir = Path(cfg["output_dir"]) / cfg["run_name"]; create it
    #      (run_dir.mkdir(parents=True, exist_ok=True)) and save the config into it as
    #      config.json - every run must remember exactly which settings produced it!
    #   e) print the stage, device and run_dir
    #   f) call the right function for cfg["stage"]:
    #        "data"  -> run_data_stage(cfg, device, run_dir)
    #        "model" -> run_model_stage(cfg, device)
    #        "train" -> run_train_stage(cfg, device, run_dir)
    #        "full"  -> run_full_stage(cfg, device, run_dir)
    raise NotImplementedError("TODO [Week 1] 6: main")


if __name__ == "__main__":
    main()
