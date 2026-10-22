"""Week 3 - Evaluation: loss/accuracy, predictions, reports, plots, and run comparison.

Compare finished runs from the command line:
    python evaluation.py runs/aug_none runs/aug_flip runs/aug_crop_flip --out runs/aug.png

Complete every block marked TODO. Read README.md of Week 3 first.
Suggested order:
    1. evaluate     3. report                  5. plot_history     7. load_run + compare_runs
    2. predict      4. plot_confusion_matrix   6. summarize
Also finish the "TODO [Week 3]" blocks in main.py, then start the fine-tuning lab (README).

Useful documentation:
    sklearn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from resnet import soft_cross_entropy


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean loss, accuracy) without updating the model."""
    # TODO 1: like resnet.train_one_epoch, but:
    #   - model.eval() instead of model.train()   (dropout OFF, BatchNorm uses running stats)
    #   - NO zero_grad / backward / step           (we only measure, never learn here)
    #   - @torch.no_grad() above already disables gradient tracking (faster, less memory)
    #   Return (total_loss / seen, correct / seen).
    raise NotImplementedError("TODO 1: evaluate")


@torch.no_grad()
def predict(model, loader, device):
    """Returns (true labels, predicted labels) as integer numpy arrays."""
    # TODO 2: model.eval(); for every batch collect
    #     predicted class = logits.argmax(1)  (move to CPU with .cpu())
    #     true class      = targets.argmax(1) (one-hot -> class index)
    #   in two lists, then return torch.cat(list).numpy() for both.
    raise NotImplementedError("TODO 2: predict")


def report(y_true, y_pred, class_names, save_path=None):
    """Per-class precision / recall / F1 table. Prints it, optionally saves it, returns it."""
    # TODO 3: text = classification_report(y_true, y_pred, target_names=class_names,
    #                                      digits=4, zero_division=0)
    #   print it, write it to save_path if given (Path(save_path).write_text(text)), return it.
    raise NotImplementedError("TODO 3: report")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=True):
    """normalize=True shows per-class recall (each row sums to 1)."""
    # TODO 4: compute
    #   cm    = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    #           cm[i, j] = number of images of TRUE class i that were PREDICTED as class j
    #   shown = cm divided by its row sums (keepdims=True) if normalize, else cm
    #           (guard against division by zero with np.maximum(row_sums, 1))
    #   Question: why does a row-normalized matrix show RECALL and not precision?
    raise NotImplementedError("TODO 4: plot_confusion_matrix")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(shown, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title="Confusion matrix" + (" (row-normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # TODO 4b: write the number into every cell: for every (i, j) call
    #   ax.text(j, i, value, ha="center", va="center", fontsize=8, color=...)
    #   value = f"{shown[i, j]:.2f}" if normalize else str(cm[i, j])
    #   color = "white" on dark cells (shown[i, j] > shown.max() / 2), "black" otherwise.
    #   Note: ax.text takes (x=column, y=row)!

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return cm


def plot_history(history, save_path):
    """Loss, accuracy and generalization-gap curves; the dashed line marks the best epoch.

    Three side-by-side panels: "Loss", "Accuracy", "Generalization gap (train acc - val acc)".
    """
    # TODO 5:
    #   a) epochs = np.arange(1, len(history["train_loss"]) + 1)
    #      fig, (ax_loss, ax_acc, ax_gap) = plt.subplots(1, 3, figsize=(16, 4.5))
    #   b) plot train_loss on ax_loss and train_acc on ax_acc (label="train")
    #   c) only if history["val_acc"] is not empty (stage=train has no validation!):
    #        - plot val_loss / val_acc (label="val")
    #        - gap = np.array(train_acc) - np.array(val_acc), plot it on ax_gap
    #        - best = np.argmax(val_acc) + 1; draw
    #          ax.axvline(best, color="gray", linestyle="--", label=f"best epoch ({best})")
    #          on all three axes
    #   d) titles, xlabel="Epoch", ax.grid(alpha=0.3), ax.legend() on every axis
    #   e) fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
    raise NotImplementedError("TODO 5: plot_history")


def summarize(history, test_loss, test_acc):
    """Key numbers of one run, saved as results.json and used by compare_runs."""
    # TODO 6: return a dict with exactly these keys (convert numpy numbers with float()/int()
    #   so json can save them). "best" = the epoch with the highest validation accuracy.
    #     best_epoch          (1-based)
    #     best_val_acc
    #     train_acc_at_best
    #     gap_at_best         train_acc_at_best - best_val_acc
    #     final_train_acc     (last epoch)
    #     final_val_acc       (last epoch)
    #     final_gap
    #     min_val_loss_epoch  (1-based epoch with the LOWEST val_loss)
    #     test_loss, test_acc
    #     train_minutes       sum(history["epoch_time"]) / 60
    #   Think: why can best_epoch and min_val_loss_epoch be different?
    raise NotImplementedError("TODO 6: summarize")


def load_run(run_dir):
    """Returns {"name": folder name, "config": {...}, "history": {...}, "results": {...}}."""
    # TODO 7a: read config.json, history.json and results.json of run_dir with json.loads.
    #   A missing file becomes {} (e.g. a run that crashed or is still training).
    raise NotImplementedError("TODO 7a: load_run")


def compare_runs(run_dirs, save_path=None):
    """Print a table of finished runs (sorted by validation accuracy) and plot their curves."""
    # TODO 7a (continued): runs = load_run(...) for every folder in run_dirs, but keep only
    #   the runs that have results (finished). If none are left, print a message and return [].

    # TODO 7b: find the config keys whose value is NOT the same in all runs
    #   (ignore "run_name") and print them. Store them in the list `differing`.
    #   Hint: collect all keys of all r["config"]; a key differs if the set of its values
    #   (use json.dumps(value) to make them hashable) has more than one element.

    # TODO 7c: sort `runs` by r["results"]["best_val_acc"], highest first.
    #   IMPORTANT: sort by VALIDATION accuracy, NOT test accuracy. Why? (README, data leakage)

    # TODO 7d: print one table row per run with these columns:
    #     run | best val | test | train@best | gap@best | best ep | minutes | settings
    #   where "settings" shows only the differing keys, e.g. "augment=none dropout=0.3".
    #   Tip: f-strings with fixed widths line the columns up: f"{name:24s} {acc:9.4f}"
    raise NotImplementedError("TODO 7b-7d: compare_runs table")

    if save_path:
        fig, (ax_acc, ax_loss, ax_gap) = plt.subplots(1, 3, figsize=(17, 4.8))
        for r in runs:
            h = r["history"]
            epochs = np.arange(1, len(h["val_acc"]) + 1)
            # TODO 7e: draw this run in ONE color on all three panels:
            #   line, = ax_acc.plot(epochs, h["val_acc"], label=r["name"])   (solid = val)
            #   then, with color=line.get_color():
            #   - h["train_acc"] on ax_acc (alpha=0.35, linestyle=":")       (dotted = train)
            #   - h["val_loss"] on ax_loss (label=r["name"])
            #   - the gap train_acc - val_acc on ax_gap (label=r["name"])
        ax_acc.set(title="Accuracy (solid = val, dotted = train)", xlabel="Epoch")
        ax_loss.set(title="Validation loss", xlabel="Epoch")
        ax_gap.set(title="Generalization gap (train - val acc)", xlabel="Epoch")
        for ax in (ax_acc, ax_loss, ax_gap):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"Saved comparison plot to {save_path}")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare finished training runs.")
    parser.add_argument("runs", nargs="+", help="run directories, e.g. runs/baseline runs/no_aug")
    parser.add_argument("--out", default=None, help="where to save the comparison plot (.png)")
    args = parser.parse_args()
    run_dirs = sorted({path for pattern in args.runs for path in (glob.glob(pattern) or [pattern])})
    compare_runs(run_dirs, args.out)
