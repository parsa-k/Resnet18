# Week 3 — Evaluation and the Fine-Tuning Lab (`evaluation.py`)

> **Goal of this week:** measure your model *honestly*, then run a series of controlled experiments
> to discover what really drives accuracy: **overfitting, data leakage, data augmentation,
> dropout and other regularizers, and hyperparameter tuning.**
>
> By the end you should be able to look at a pair of training curves and say
> *what is going wrong and which knob to turn*.

**Files this week:** copy `evaluation.py` into your working folder. Complete all its TODOs and the
`TODO [Week 3]` blocks (9, 11-14) in `main.py`.
**Commands you will run:** `python main.py stage=full ...` and `python evaluation.py runs/... --out ...`

---

## Part A — Background

### A.1 Evaluating a classifier

`model.eval()` switches dropout off and makes BatchNorm use its stored running statistics, so predictions are
deterministic. `torch.no_grad()` skips gradient bookkeeping, which saves memory and time. **Never** call `backward()` or
`optimizer.step()` on validation or test data.

For every class *c*, count the predictions:
TP = predicted *c* and really *c*; FP = predicted *c* but not *c*; FN = really *c* but predicted something else.

| Metric | Formula | Question it answers |
|--------|---------|---------------------|
| Accuracy | correct / all | How often is the model right overall? |
| Precision (class *c*) | TP / (TP + FP) | When the model says "cat", how often is it really a cat? |
| Recall (class *c*) | TP / (TP + FN) | Of all real cats, how many did the model find? |
| F1 (class *c*) | 2·P·R / (P + R) | One number balancing precision and recall |
| Macro average | mean over classes | Every class counts equally |

The **confusion matrix** `cm[i, j]` counts images of true class *i* predicted as class *j*.
The diagonal holds the correct predictions; off-diagonal cells show *which* classes get confused (on CIFAR-10, typically cat ↔ dog and automobile ↔ truck).
Dividing each row by its sum gives the **recall** of each class on the diagonal.

### A.2 Learning curves, underfitting and overfitting

We record, for every epoch, the loss and accuracy on the training set *and* on the validation set.
The **generalization gap** = train accuracy − validation accuracy.

| Situation | Training accuracy | Validation accuracy | Gap | Typical fix |
|-----------|-------------------|---------------------|-----|-------------|
| **Underfitting** (high bias) | low | low | small | bigger model, train longer, higher LR, less regularization |
| **Good fit** | high | high | small | — |
| **Overfitting** (high variance) | very high (→ 100 %) | much lower, stops improving or drops | large and growing | more data, augmentation, weight decay, dropout, early stopping, smaller model |

Overfitting means the model **memorizes** the training images (including their noise and peculiarities) instead
of learning patterns that transfer to new images. A ResNet-18 has 11 million parameters and can memorize 45,000 images
perfectly, so on CIFAR-10 overfitting is the main enemy.

A typical sign: **validation loss goes up while validation accuracy stays flat.** The model becomes *over-confident*: on images it gets wrong,
it assigns a tiny probability to the true class, and $-\log p$ explodes.

### A.3 Early stopping and the best checkpoint

Validation accuracy usually peaks somewhere and then stays flat or drops. Our `train` function saves the weights
from the epoch with the **best validation accuracy** (`best_model.pt`) and the test set is evaluated with *those*
weights, not the last ones. This is **early stopping**: one of the simplest and most effective regularizers.

### A.4 Data leakage

> **Data leakage** = information that would not be available when the model is used for real
> finds its way into training or into model selection. The result: **evaluation numbers that are too optimistic**,
> often dramatically, and a model that disappoints in practice.

Common forms:

| Form | Example | In this project |
|------|---------|-----------------|
| Train/validation overlap | the same (or near-duplicate) images in both sets | `leak=val_in_train` |
| Model selection on the test set | picking the best epoch / hyperparameters by test accuracy | `leak=test_as_val`, and choosing runs by test accuracy |
| Preprocessing fitted on all data | normalization mean/std, PCA, vocabulary... computed on train + test | we compute statistics on training images only (Week 1) |
| Repeated "peeking" | trying 100 settings and reporting the best test score | the "winner's curse" in Lab C3 |
| Temporal / group leakage | future data used to predict the past; same patient in train and test | (not in CIFAR, but very common in practice) |

**The golden rule:** decide *everything* (epochs, learning rate, augmentation, dropout, architecture) using the
**validation set**. Touch the test set **once**, at the very end, to report the final number.

### A.5 Data augmentation as regularization

Augmentation (Week 1) makes every epoch show slightly different images. The model can no longer memorize exact pixels,
so it must learn features that are robust to shifts, flips and colour changes. Expected effects:

* **training accuracy goes down** (the task is harder) — this is fine!
* **validation accuracy goes up** and the gap shrinks;
* too strong augmentation for a short training run can cause **underfitting** — the model needs more epochs to benefit.

### A.6 Dropout, weight decay and label smoothing

**Dropout** (Srivastava et al., 2014): during training, each feature is set to 0 with probability *p*, and the survivors
are multiplied by $1/(1-p)$ so the expected value stays the same ("inverted dropout"). At evaluation time, nothing is dropped.

* The network cannot rely on any single feature → more redundant, robust representations.
* It can be seen as training an *ensemble* of many thinned networks that share weights.
* In our ResNet-18, dropout sits only before the final `fc` layer, so its effect is milder than in the large fully
  connected networks it was invented for. Modern CNNs rely more on BatchNorm, augmentation and weight decay. Measure it yourself!

**Weight decay** adds $\lambda \lVert \theta \rVert^2$ to the loss, pulling weights towards zero, which favours simpler functions.
(With BatchNorm the full story is more subtle: weight decay also increases the effective learning rate.)

**Label smoothing** (Week 1) penalizes over-confident predictions, which often improves validation loss and calibration.

### A.7 Hyperparameter tuning

*Parameters* (weights) are learned by gradient descent. *Hyperparameters* (learning rate, batch size, epochs,
weight decay, dropout, augmentation, width...) are chosen by **you**, by comparing runs on the **validation set**.

Good practice:

1. **Start from a known good baseline** (here: SGD, lr 0.1, momentum 0.9, weight decay 5e-4, cosine schedule, crop+flip).
2. **Change one thing at a time** and keep the seed fixed, otherwise you cannot tell what caused a change.
3. **Search on a log scale** for learning rate and weight decay: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3...
4. **Coarse-to-fine:** a wide, cheap search first (small subset, few epochs), then a narrow search around the best value.
5. **Random search beats grid search** when only a few hyperparameters really matter (Bergstra & Bengio, 2012).
6. **Know your noise level:** train the same configuration with 2-3 seeds. A difference smaller than the seed-to-seed
   variation is *not* a real improvement.
7. **Hyperparameters interact:** the best learning rate depends on the batch size and the optimizer
   (Adam wants ~1e-3, SGD ~0.1); the best dropout depends on how much augmentation you use.
8. **Final model:** retrain the chosen configuration (optionally on more data / longer), then evaluate on the test set **once**.

---

## Part B — Your tasks

**`evaluation.py`**

1. `evaluate` — loss and accuracy without learning
2. `predict` — true and predicted class indices
3. `report` — precision / recall / F1 per class (scikit-learn)
4. `plot_confusion_matrix` — compute and annotate the matrix
5. `plot_history` — loss, accuracy and generalization-gap curves with the best epoch marked
6. `summarize` — the key numbers of a run (saved as `results.json`)
7. `load_run` + `compare_runs` — a table and plots that compare several runs

**`main.py`** — `TODO [Week 3]` blocks: 9 (validation + checkpointing inside `train`), 11-14 (`run_full_stage`).

---

## Part C — Step-by-step checks

```bash
# 1-2, 4. evaluate / predict / confusion matrix with a fake "model" that returns its input as logits
python -c "
import torch, evaluation
from torch.utils.data import TensorDataset, DataLoader
x = torch.tensor([[5., 0., 0.], [0., 5., 0.], [5., 0., 0.], [0., 0., 5.]])
y = torch.eye(3)[[0, 1, 1, 2]]
loader = DataLoader(TensorDataset(x, y), batch_size=2)
model = torch.nn.Identity()
print(evaluation.evaluate(model, loader, 'cpu'))
print(evaluation.predict(model, loader, 'cpu'))
print(evaluation.plot_confusion_matrix(*evaluation.predict(model, loader, 'cpu'), ['a', 'b', 'c'], 'cm_test.png'))"
# (1.2633858378976583, 0.75)
# (array([0, 1, 1, 2]), array([0, 1, 0, 2]))
# [[1 0 0]
#  [1 1 0]
#  [0 0 1]]

# 5-6. summarize and plot_history on a fake 3-epoch history
python -c "
import evaluation
h = {'train_loss': [1.4, 0.6, 0.2], 'train_acc': [0.5, 0.8, 0.95], 'val_loss': [1.5, 1.2, 1.3],
     'val_acc': [0.45, 0.6, 0.58], 'epoch_time': [60, 60, 60]}
print(evaluation.summarize(h, 1.25, 0.59))
evaluation.plot_history(h, 'history_test.png')"
# {'best_epoch': 2, 'best_val_acc': 0.6, 'train_acc_at_best': 0.8, 'gap_at_best': 0.2000..., 'final_train_acc': 0.95,
#  'final_val_acc': 0.58, 'final_gap': 0.37, 'min_val_loss_epoch': 2, 'test_loss': 1.25, 'test_acc': 0.59, 'train_minutes': 3.0}
```

Then a short end-to-end run:

```bash
python main.py stage=full subset_fraction=0.1 width=16 epochs=3 run_name=check
```

`runs/check/` must now contain `config.json`, `history.json`, `best_model.pt`, `training_curves.png`,
`classification_report.txt`, `confusion_matrix.png` and `results.json`. Then:

```bash
python evaluation.py runs/check --out runs/check_compare.png
```

---

## Part D — The Fine-Tuning Lab

### D.0 Lab settings and workflow

Full ResNet-18 runs are slow, so the lab uses a **small, fast setting**: 10 % of the training data
(4,500 train / 500 val images), width 16 (0.7 M parameters), 20 epochs. That takes about **4-5 minutes per run on a laptop CPU**
and under a minute on a GPU. The *trends* you will see are the same as for the full model.

Define a shortcut once per terminal session:

```bash
# macOS / Linux (bash or zsh)
lab() { python main.py stage=full subset_fraction=0.1 width=16 epochs=20 "$@"; }
```

```powershell
# Windows PowerShell
function lab { python main.py stage=full subset_fraction=0.1 width=16 epochs=20 @args }
```

Anything you add overrides the defaults, e.g. `lab augment=none run_name=B1_no_reg`.

The loops below use bash/zsh syntax. In PowerShell write them as
`foreach ($f in 0.05, 0.1, 0.2) { lab subset_fraction=$f run_name=B2_size_$f }`
and put patterns in quotes: `python evaluation.py "runs/B2_size_*"`.

After each part, compare its runs:

```bash
python evaluation.py runs/A_baseline runs/B1_no_reg --out runs/compare_B.png
```

The table is sorted by **best validation accuracy** and shows train accuracy, gap, test accuracy, best epoch and
the settings that differ between the runs. For each part, put the table, the comparison figure and your answers in your report.

> **Rule for the whole lab:** make every decision by looking at `best val`. Look at `test` only to *discuss*
> the results afterwards — never to choose. (Part C explains why.)

---

### Lab A — Baseline

```bash
lab run_name=A_baseline
```

Open `runs/A_baseline/training_curves.png`, `confusion_matrix.png` and `classification_report.txt`.

* A.1 What are the best validation accuracy, the test accuracy and the best epoch?
* A.2 Which two classes are confused most often? Show the cell in the confusion matrix. Why are they hard to tell apart at 32×32 pixels?
* A.3 Which class has the lowest recall, and which has the lowest precision? Explain the difference in words, using those classes as the example.

---

### Lab B — Overfitting

**B1. Remove all regularization** (no augmentation, no weight decay):

```bash
lab augment=none weight_decay=0 run_name=B1_no_reg
python evaluation.py runs/A_baseline runs/B1_no_reg --out runs/compare_B1.png
```

* B1.1 What is the training accuracy at the end? The validation accuracy? The gap? Compare with the baseline.
* B1.2 In `training_curves.png` of B1, find the epoch where the **validation loss** is lowest
  (`min_val_loss_epoch` in `results.json`) and the epoch where **validation accuracy** is highest (`best_epoch`).
  Are they the same? Why does the validation loss *rise* while validation accuracy stays roughly constant?
* B1.3 Was saving the best checkpoint (early stopping) helpful here? How much worse would the *last* epoch have been?

**B2. Effect of the amount of data** (the 0.4 run takes ~4× longer; optional on CPU):

```bash
for f in 0.05 0.1 0.2 0.4; do lab augment=none weight_decay=0 subset_fraction=$f run_name=B2_size_$f; done
python evaluation.py runs/B2_size_* --out runs/compare_B2.png
```

* B2.1 Plot (or tabulate) best validation accuracy and gap against the number of training images.
* B2.2 More data is the best cure for overfitting. Explain why, in terms of memorization.

**B3. Effect of model size** (optional):

```bash
for w in 8 16 32; do lab augment=none weight_decay=0 width=$w run_name=B3_width_$w; done
```

* B3.1 Does a bigger model overfit more, or less? Does it still get a higher validation accuracy? (The answer may surprise you.)

---

### Lab C — Data leakage

Use B1's settings (no regularization) so the effect is easy to see.

**C1. Validation images leaked into the training set:**

```bash
lab augment=none weight_decay=0 leak=val_in_train run_name=C1_val_in_train
python evaluation.py runs/B1_no_reg runs/C1_val_in_train --out runs/compare_C1.png
```

* C1.1 Compare `best val` and `test` of both runs. What does the leaked validation accuracy suggest, and what is the truth?
* C1.2 Look at the comparison plot: in C1, why does the validation curve follow the training curve?
* C1.3 Imagine you had used C1's validation accuracy to tune dropout or decide when to stop. Which values would you have chosen, and why would they be wrong?
* C1.4 Real-world version: CIFAR-10's test set contains a few percent of images that are near-duplicates of training images
  (Barz & Denzler, *"Do we train on test data?"*, 2020). How could near-duplicates distort reported results, even when nobody makes a coding mistake?

**C2. Picking the best epoch with the test set:**

```bash
lab augment=none weight_decay=0 leak=test_as_val run_name=C2_test_as_val
python evaluation.py runs/B1_no_reg runs/C2_test_as_val
```

* C2.1 In C2 the "validation" numbers *are* test numbers. Compare C2's reported test accuracy with B1's. Which one is an honest estimate and why?
* C2.2 The difference is probably small here. Why does it grow when you make *many* decisions using the test set?

**C3. The winner's curse (do this after Lab F):** take all your Lab F runs.

* C3.1 Which run would you pick by **best val**? Which by **test**? Are they the same?
* C3.2 For the run with the highest test accuracy, is its test accuracy a fair estimate of how it will perform on new data? Explain.

**C4. Preprocessing leakage (thought experiment, no code):** in `load_cifar10` we compute the normalization statistics with
`raw.data[train_idx]`. Describe what would change if we used all 60,000 images, including the test set. Why is
the effect small for mean/std, but dangerous for other preprocessing (e.g. feature selection)?

---

### Lab D — Data augmentation

```bash
for a in none flip crop_flip strong; do lab augment=$a run_name=D1_aug_$a; done
python evaluation.py runs/D1_aug_* --out runs/compare_D1.png
```

Strong augmentation often needs longer training to pay off:

```bash
lab augment=crop_flip epochs=40 run_name=D2_crop_flip_40ep
lab augment=strong    epochs=40 run_name=D2_strong_40ep
```

* D.1 Fill in the table: augmentation → final train acc, best val acc, gap. How does the train accuracy change, and why is that *not* bad?
* D.2 Which augmentation gives the best validation accuracy after 20 epochs? After 40?
* D.3 Look back at your Week 1 augmentation pictures. Relate what you see to these numbers.
* D.4 Is there a sign of *underfitting* in any run? Which curve tells you?

---

### Lab E — Dropout and other regularizers

**E1. Dropout without other regularization** (the setting where it can help most):

```bash
for p in 0.0 0.2 0.5 0.8; do lab augment=none weight_decay=0 dropout=$p run_name=E1_drop_$p; done
python evaluation.py runs/E1_drop_* --out runs/compare_E1.png
```

**E2. Dropout on top of augmentation + weight decay:**

```bash
for p in 0.0 0.5; do lab dropout=$p run_name=E2_aug_drop_$p; done
```

**E3. Weight decay and E4. label smoothing:**

```bash
for wd in 0 0.0005 0.005; do lab augment=none weight_decay=$wd run_name=E3_wd_$wd; done
for ls in 0.0 0.1; do lab label_smoothing=$ls run_name=E4_ls_$ls; done
```

* E.1 How does the training accuracy change as dropout increases? The validation accuracy? The gap?
* E.2 Is dropout more or less useful when augmentation is already used (E1 vs E2)? Why might regularizers "overlap"?
* E.3 Our dropout is applied only to the 8 × width pooled features just before `fc` (128 features for width 16, 512 for width 64). Why might the effect be small? Where else could you put dropout? (Optional: try `nn.Dropout2d` inside `BasicBlock`.)
* E.4 Rank the regularizers you tried (augmentation, weight decay, dropout, label smoothing, early stopping) by how much they helped **validation accuracy** in this setting.
* E.5 With label smoothing, compare the validation **loss** of E4 runs. Why does label smoothing change the loss values a lot even when accuracy barely moves?

---

### Lab F — Hyperparameter tuning

**F1. Learning rate (SGD), log scale:**

```bash
for lr in 0.003 0.01 0.03 0.1 0.3 1.0; do lab lr=$lr run_name=F1_lr_$lr; done
python evaluation.py runs/F1_lr_* --out runs/compare_F1.png
```

* F1.1 Plot best val accuracy against learning rate (log x-axis). Where is the optimum? What happens at both ends?
* F1.2 Coarse-to-fine: pick the two best values and try 2-3 values between them. Did it help?

**F2. Optimizers** (each has its own good learning rate):

```bash
lab optimizer=sgd   lr=0.1                     run_name=F2_sgd
lab optimizer=sgd   lr=0.1 momentum=0          run_name=F2_sgd_nomomentum
lab optimizer=adam  lr=0.001                   run_name=F2_adam
lab optimizer=adamw lr=0.001 weight_decay=0.05 run_name=F2_adamw
lab optimizer=adam  lr=0.1                     run_name=F2_adam_lr0.1
```

* F2.1 Which optimizer converges fastest in the first epochs? Which reaches the best validation accuracy?
* F2.2 What happens to Adam with SGD's learning rate (0.1)?

**F3. Batch size** (and the "linear scaling rule": scale lr with the batch size):

```bash
lab batch_size=32  run_name=F3_bs32
lab batch_size=512 run_name=F3_bs512
lab batch_size=32  lr=0.025 run_name=F3_bs32_lr0.025
lab batch_size=512 lr=0.4   run_name=F3_bs512_lr0.4
```

* F3.1 Compare time per epoch and best val accuracy. Why does a larger batch give fewer parameter updates per epoch?
* F3.2 Did scaling the learning rate with the batch size help?

**F4. Learning-rate schedule and F5. training length:**

```bash
for s in none step linear cosine; do lab scheduler=$s run_name=F4_sched_$s; done
for e in 10 20 40; do lab epochs=$e run_name=F5_epochs_$e; done
```

* F4.1 Use the `lr` entries in `history.json` to explain the shape of each validation curve.
* F5.1 Is training longer always better? Look at the gap.

**F6. Random search (optional):** sample learning rate and weight decay on a log scale.

```bash
for i in 1 2 3 4 5 6; do
  lr=$(python -c "import random; print(round(10 ** random.uniform(-2.5, -0.3), 4))")
  wd=$(python -c "import random; print(f'{10 ** random.uniform(-5, -2.5):.1e}')")
  lab lr=$lr weight_decay=$wd run_name=F6_random_$i
done
python evaluation.py runs/F6_random_* --out runs/compare_F6.png
```

**F7. Noise level — how big is a "real" difference?**

```bash
for s in 1 2 3; do lab seed=$s run_name=F7_seed_$s; done
python evaluation.py runs/F7_seed_*
```

* F7.1 What is the spread (max − min) of best val accuracy across seeds? Go back to Labs D-F: which of your
  "improvements" are larger than this spread, and which could just be noise?
* F7.2 Why does a 500-image validation set make this noise large? (Recall Week 1, Experiment 4.)

---

### Lab G — Architecture ablations (optional)

```bash
lab residual=False   run_name=G_plain
lab stem=imagenet    run_name=G_imagenet_stem
lab width=32         run_name=G_width32
python evaluation.py runs/A_baseline runs/G_* --out runs/compare_G.png
```

* G.1 Check your predictions from Week 2: did the residual connections and the CIFAR stem help? By how much?

---

### Lab H — Your final model

1. Combine your best choices from Labs D-G (chosen on **validation** accuracy only!).
2. Train it on all data with more capacity if your hardware allows, for example:

```bash
python main.py stage=full subset_fraction=1.0 width=64 epochs=30 <your best settings> run_name=H_final
```

   (On a CPU-only machine, use `subset_fraction=0.5 width=32 epochs=20`, or use Colab.)
3. Report its test accuracy **once**. Do not tune anything afterwards.

* H.1 How much did the test accuracy improve compared with the lab baseline? Which change mattered most?
* H.2 Show the final confusion matrix. Which classes are still hard?
* H.3 For reference, a well-tuned ResNet-18 trained on all data for ~200 epochs reaches ~95 %. What is still missing in your run?

---

## Part E — Report template

Write a short report (3-6 pages, figures included):

1. **Setup** — hardware, lab settings, time per run.
2. **Baseline** — numbers, curves, confusion matrix (Lab A).
3. **Overfitting** — table + figure (Lab B), your diagnosis of the curves.
4. **Data leakage** — table (Lab C), explain in your own words why the leaked numbers are misleading.
5. **Augmentation and regularization** — tables (Labs D, E), ranking of regularizers.
6. **Hyperparameter tuning** — LR curve, optimizer/batch/schedule tables, seed noise (Lab F).
7. **Final model** — chosen settings and *why*, final test accuracy (Lab H).
8. **Lessons learned** — five bullet points you would tell a classmate starting this project.

A results table format you can reuse:

| Run | Changed setting | Best val acc | Test acc | Train acc @ best | Gap | Best epoch |
|-----|-----------------|--------------|----------|------------------|-----|------------|
| A_baseline | — | | | | | |
| B1_no_reg | augment=none, weight_decay=0 | | | | | |
| ... | | | | | | |

---

## Part F — Reference numbers (open only AFTER running the experiments)

<details>
<summary>Instructor's results with the lab settings (CPU, seed 42) — your numbers will differ slightly</summary>

Lab settings: `subset_fraction=0.1 width=16 epochs=20`, about 5 minutes per run on a laptop CPU.

| Run | Changed setting | Best val acc | Test acc | Train acc @ best | Gap @ best | Best epoch |
|-----|-----------------|--------------|----------|------------------|------------|------------|
| A_baseline | — (crop_flip, wd 5e-4, SGD lr 0.1) | 0.626 | 0.623 | 0.718 | 0.092 | 19 |
| B1_no_reg | augment=none, weight_decay=0 | 0.508 | 0.507 | 0.988 | 0.480 | 13 |
| C1_val_in_train | B1 + leak=val_in_train | **1.000** | 0.524 | 0.982 | −0.018 | 15 |
| E1_drop_0.5 | B1 + dropout=0.5 | 0.478 | 0.487 | 0.980 | 0.502 | 16 |
| F1_lr_0.01 | lr=0.01 | 0.502 | 0.511 | 0.538 | 0.036 | 20 |
| F2_adam | optimizer=adam, lr=0.001 | 0.594 | 0.600 | 0.655 | 0.061 | 17 |
| G_plain | residual=False | 0.526 | 0.553 | 0.611 | 0.085 | 19 |

These runs show the main lessons of the lab:

* **Overfitting (B1):** without regularization the model memorizes the training set (99 % train) but only reaches about 51 % on validation, a gap of almost 50 points.
  The validation loss is lowest within the first few epochs and then rises, while validation accuracy still creeps up and then plateaus: the model grows over-confident.
* **Data leakage (C1):** once the validation images are part of the training set, validation accuracy reads **100 %**,
  yet the test accuracy stays at about 52 %. The validation number is worthless.
* **Augmentation + weight decay (A):** train accuracy drops to about 72 %, but validation and test rise to about 62 % and the gap shrinks to about 9 points.
* **Dropout (E1):** in this setting dropout 0.5 before `fc` did *not* help (it was slightly worse). With only 500 validation images,
  a 2-3 point difference is close to the seed noise (Lab F7). Augmentation is far more effective for a CNN.
* **Learning rate (F1):** lr 0.01 is too small for 20 epochs. The model **underfits** (train accuracy is only 54 %, the gap is tiny).
* **Optimizer (F2):** Adam with its default lr works well out of the box, but tuned SGD + momentum is better here.
* **Residual connections (G):** removing the skip connections cost about 10 points of validation accuracy, even at only 18 layers.



</details>
