# Step 1 — The Data Pipeline (`data.py` + `main.py`)

> **Goal of this week:** turn 60,000 tiny colour images into clean, normalized,
> correctly labelled **batches of tensors** that a neural network can learn from —
> and *see* every transformation you apply.
>
> "Garbage in, garbage out": most bugs in deep learning projects are data bugs.
> A perfect model trained on a broken pipeline still fails.

**Files this week:** `main.py` (only the `TODO [Week 1]` blocks) and `data.py` (all TODOs).
**Command you will run:** `python main.py stage=data ...`

---

## Part A — Background

### A.1 Image classification and CIFAR-10

*Image classification* means: given an image, predict one label out of a fixed set of classes.
We learn the mapping from **examples** (supervised learning): many images, each with its correct label.

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a classic benchmark:


| Property       | Value                                                                    |
| -------------- | ------------------------------------------------------------------------ |
| Images         | 60,000 colour images, **32 × 32 pixels**, 3 channels (RGB)               |
| Classes        | 10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Official split | 50,000 training + 10,000 test images                                     |
| Balance        | exactly 6,000 images per class (5,000 train + 1,000 test)                |


32 × 32 is tiny — even humans make mistakes (is that blob a cat or a dog?).
A good ResNet-18 reaches ~93-95 % test accuracy; random guessing gives 10 %.

### A.2 Images as tensors

On disk (and in `datasets.CIFAR10(...).data`) an image is a **NumPy array of shape** `(H, W, C)` **=**
`(32, 32, 3)` **with integers 0-255** (`uint8`). PyTorch convolutions expect `(C, H, W)` **floats**.
`transforms.ToTensor()` does both conversions:

```
uint8  (32, 32, 3)  values 0..255     --ToTensor-->     float32  (3, 32, 32)  values 0.0..1.0
```

A **batch** stacks many images: `(B, 3, 32, 32)`, e.g. `(128, 3, 32, 32)`.

### A.3 Dataset, transforms and DataLoader

PyTorch separates three jobs:


| Object                           | Job                                                   | In our code                  |
| -------------------------------- | ----------------------------------------------------- | ---------------------------- |
| `Dataset`                        | "give me sample number *i*" → `(image, target)`       | `datasets.CIFAR10`, `Subset` |
| `transform` / `target_transform` | change a sample on the fly every time it is read      | `build_transforms`, `OneHot` |
| `DataLoader`                     | group samples into batches, shuffle, load in parallel | `make_loaders`               |


Important details:

- **Transforms run every time a sample is read.** A *random* transform (like a random flip) therefore gives a
*different* version of the same image in every epoch. This is how augmentation works.
- `shuffle=True` for training: if the network always saw all airplanes first, then all cars…,
each gradient step would be biased towards one class. Validation/test do not need shuffling —
the result is the same in any order.
- `num_workers` = number of parallel processes preparing batches while the GPU computes.
Too few → the GPU waits for data. `0` = load in the main process (simplest, good for debugging).
- `Subset(dataset, indices)` is a "view" that only exposes the given indices — no copying.



### A.4 Train / validation / test — why three sets?


| Set            | Used for                                                                                  | Rule                          |
| -------------- | ----------------------------------------------------------------------------------------- | ----------------------------- |
| **Training**   | learning the weights (gradient descent)                                                   | the model sees it every epoch |
| **Validation** | choosing *anything* you choose by hand: best epoch, learning rate, augmentation, dropout… | never trained on              |
| **Test**       | **one** final, honest estimate of performance on new data                                 | never used for any decision   |


CIFAR-10 only ships train and test, so we carve a validation set out of the training
images (`val_split=0.1` → 45,000 train / 5,000 val).

If information from the validation or test set influences training, your accuracy
numbers become **too optimistic**. This is called **data leakage**. You will measure exactly how
misleading it is in Week 3 (`leak=...` setting). This week you write the check that detects one
form of it: **train/val overlap must be 0**.

### A.5 Normalization

Raw pixel values are all positive (0…1) and the three channels have different averages.
We standardize each channel *c*:

$$
x' = \frac{x - \mu_c}{\sigma_c}
$$

so every channel has **mean ≈ 0 and standard deviation ≈ 1**. Why this helps:

- inputs centred around 0 give better-conditioned gradients → faster, more stable training;
- no channel dominates just because its numbers are larger.

The statistics $\mu_c, \sigma_c$ are computed **on the training images only**. The validation and
test images are normalized with *those same* numbers — in real life you do not know the future data in advance.
Computing them on the test set is (a mild) data leakage.

For the whole CIFAR-10 training set: $\mu \approx (0.491, 0.482, 0.447)$, $\sigma \approx (0.247, 0.243, 0.262)$.

### A.6 One-hot encoding and label smoothing

The label of an image is an integer (e.g. `3` = cat). The network outputs 10 numbers (one per class),
so we write the target as a vector of 10 probabilities:

```
label 3   -->   one-hot  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

**Label smoothing** with factor ε moves a little probability mass to the other classes:

$$
y_{\text{smooth}} = (1-\varepsilon)\, y_{\text{one-hot}} + \frac{\varepsilon}{K}, \qquad K = 10
$$

With ε = 0.1: `0.91` for the true class and `0.01` for each of the others (still sums to 1).
It tells the model "don't be 100 % sure" — a regularizer you will test in Week 3.
Smoothing is applied **only to training targets**; validation/test targets stay hard one-hot vectors.

### A.7 Data augmentation

Augmentation creates *new, plausible* training images from existing ones by random transformations
that do **not change the label**. A horse flipped left-right is still a horse.


| Level (`augment=`) | Transforms                                    | Idea                                                                        |
| ------------------ | --------------------------------------------- | --------------------------------------------------------------------------- |
| `none`             | —                                             | the model sees exactly the same 45,000 images each epoch                    |
| `flip`             | `RandomHorizontalFlip`                        | mirror image with probability 0.5                                           |
| `crop_flip`        | `RandomCrop(32, padding=4)` + flip            | pad by 4 px, cut a random 32×32 window → small shifts                       |
| `strong`           | crop + flip + `ColorJitter` + `RandomErasing` | also change brightness/contrast/saturation and black out a random rectangle |


Augmentation is used **only for training**. Validation and test images are always the clean originals —
we want to measure performance on real images, not on random distortions.
(Question: why would a *vertical* flip be a bad augmentation for CIFAR-10?)

### A.8 Reproducibility

Splitting, shuffling, augmentation and weight initialization are all random. Fixing the **seed**
(`set_seed`, `np.random.default_rng(seed)`) makes runs repeatable: the same seed → the same split →
comparable experiments. Change only *one* thing at a time, keep the seed fixed.

### A.9 Configuration and devices

`CONFIG` in `main.py` holds every setting. `parse_overrides` lets you change them from the command line
(`python main.py lr=0.01`), converting the text `"0.01"` to the type of the default value.
`get_device` picks the fastest hardware: `cuda` (NVIDIA GPU) → `mps` (Apple GPU) → `cpu`.

---



## Part B — Your tasks

Complete the TODOs in this order. After each step, run its check from Part C before moving on.

`data.py`

1. `compute_mean_std` — per-channel mean/std
2. `build_transforms` — the four augmentation levels + ToTensor + Normalize
3. `one_hot`, `OneHot.__call__` — targets as (smoothed) probability vectors
4. `split_indices` — reproducible shuffled split (with the deliberate `val_in_train` bug)
5. `load_cifar10` — puts 1-4 together; statistics from training images only
6. `make_loaders` — three DataLoaders
7. `class_counts`, `count_overlap` — sanity checks
8. `channel_stats` — verify the normalization worked
9. `benchmark_loader` — images/second
10. `describe` — print a summary
11. `save_augmentation_grid` — a picture of your augmentations

`main.py` (only the `TODO [Week 1]` blocks: 1-6)

1. `parse_overrides` 2. `set_seed` 3. `get_device` 4. `build_data` 5. `run_data_stage` 6. `main`

---



## Part C — Step-by-step checks

Run these from your working folder. Your output should match (tiny floating-point differences are fine).

```bash
# 1. compute_mean_std: one white and one black 4x4 image -> mean 0.5, std 0.5
python -c "import data, numpy as np; x = np.zeros((2,4,4,3), np.uint8); x[0] = 255; print(data.compute_mean_std(x))"
# ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# 2. build_transforms: count the transforms of each level
python -c "import data; [print(a, len(data.build_transforms(False, a)[0].transforms)) for a in data.AUGMENTATIONS]"
# none 1 / flip 2 / crop_flip 3 / strong 5

# 3. one_hot
python -c "import data; print(data.one_hot(3)); print(data.one_hot(3, smoothing=0.1)); print(data.one_hot(3, smoothing=0.1).sum())"
# tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
# tensor([0.0100, 0.0100, 0.0100, 0.9100, 0.0100, ...])
# tensor(1.)

# 4. split_indices: sizes, overlap, and the leakage bug
python -c "import data; tr, va = data.split_indices(50000, 0.1, 0.2); print(len(tr), len(va), len(set(tr) & set(va)), tr[:5])"
# 9000 1000 0 [11226 30160 20994 38831 22747]
python -c "import data; tr, va = data.split_indices(50000, 0.1, 0.2, leak='val_in_train'); print(len(tr), len(va), len(set(tr) & set(va)))"
# 10000 1000 1000

# main.py 1: parse_overrides
python -c "import main; print(main.parse_overrides({'lr': 0.1, 'augment': 'crop_flip', 'epochs': 3, 'normalize': True}, ['lr=0.01', 'epochs=5', 'normalize=false', 'augment=none']))"
# {'lr': 0.01, 'augment': 'none', 'epochs': 5, 'normalize': False}
```

Once everything is done, the full stage should run:

```bash
python main.py stage=data
```

Expected output (the image range and loader speed will differ a bit):

```
Stage: data | Device: cpu | Run dir: runs/baseline
Train: 45000 | Val: 5000 | Test: 10000
Train/val overlap: 0 images
Class counts (train): [4527, 4458, 4465, 4518, 4477, 4527, 4498, 4531, 4482, 4517]
Class counts (val):   [473, 542, 535, 482, 523, 473, 502, 469, 518, 483]
Image tensor: (3, 32, 32), dtype torch.float32, range [-1.99, 2.12]
One-hot target example: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] -> horse
Eval-set channel mean [-0.006 -0.    -0.002] | std [0.991 0.993 0.997]
Loader speed: 14,442 images/s (batch_size=128, num_workers=2)
Saved augmentation samples to runs/baseline/augmentations.png
```

Open `runs/baseline/augmentations.png`: the left column shows the original images, the other
columns show random augmented versions.

---



## Part D — Experiments: change the input, watch the output

For each experiment, run the commands, **copy the relevant output lines or figures into your report**
and answer the questions. Use a different `run_name` for each run so images are not overwritten.

### Experiment 1 — Normalization

```bash
python main.py stage=data normalize=True  run_name=w1_norm_on
python main.py stage=data normalize=False run_name=w1_norm_off
```

Look at `Image tensor ... range` and `Eval-set channel mean/std`.

- Q1.1 What is the value range with and without normalization? Why can normalized values be negative?
- Q1.2 Why is the eval-set mean close to 0 but **not exactly** 0? (Hint: which images were the statistics computed on?)
- Q1.3 Add a temporary `print(mean, std)` in `load_cifar10` and run with `subset_fraction=0.05`, `0.2` and `1.0`.
How much do the statistics change? What does that tell you about how many images you need to estimate them?



### Experiment 2 — Data augmentation (look at the pictures!)

```bash
python main.py stage=data augment=none      run_name=w1_aug_none
python main.py stage=data augment=flip      run_name=w1_aug_flip
python main.py stage=data augment=crop_flip run_name=w1_aug_crop_flip
python main.py stage=data augment=strong    run_name=w1_aug_strong
```

Compare the four `augmentations.png` files.

- Q2.1 With `augment=none`, why are all the columns identical?
- Q2.2 Where do the black borders in `crop_flip` come from?
- Q2.3 In `strong`, find one augmented image that you think is *too* hard (the object is barely visible).
Could too much augmentation hurt? (You will measure it in Week 3.)
- Q2.4 Run `augment=crop_flip` twice with the same `run_name`. Are the pictures identical? Which line of `save_augmentation_grid` causes that?



### Experiment 3 — One-hot targets and label smoothing

```bash
python main.py stage=data label_smoothing=0.0 run_name=w1_ls0
python main.py stage=data label_smoothing=0.1 run_name=w1_ls01
python main.py stage=data label_smoothing=0.5 run_name=w1_ls05
```

Look at `One-hot target example`.

- Q3.1 Write down the target vector for each value. Check that each sums to 1.
- Q3.2 What would `label_smoothing=1.0` produce? Could a model learn anything from it?
- Q3.3 Why do we **not** smooth the validation/test targets?



### Experiment 4 — Dataset size, split and class balance

```bash
python main.py stage=data subset_fraction=0.1 run_name=w1_sub01
python main.py stage=data subset_fraction=0.1 val_split=0.3 run_name=w1_sub01_val03
python main.py stage=data subset_fraction=0.02 run_name=w1_sub002
```

- Q4.1 Fill in a table: subset_fraction, val_split → #train, #val, smallest and largest class count in val.
- Q4.2 With `subset_fraction=0.02` the validation set has only 100 images. If the model gets one more image
right, how much does the validation accuracy change? Why is a tiny validation set a problem?



### Experiment 5 — Reproducibility

```bash
python main.py stage=data seed=42 run_name=w1_seed42a
python main.py stage=data seed=42 run_name=w1_seed42b
python main.py stage=data seed=7  run_name=w1_seed7
```

- Q5.1 Compare the `Class counts (val)` lines. Which runs are identical, and why?
- Q5.2 Why is it important that every experiment in Week 3 uses the same split?



### Experiment 6 — A preview of data leakage

```bash
python main.py stage=data leak=val_in_train run_name=w1_leak
```

- Q6.1 What does `Train/val overlap` show now? How many training images are there?
- Q6.2 In your own words: if the model is trained on the validation images, what will the validation accuracy
tell us about performance on *new* images? (Write down your prediction; you will test it in Week 3.)



### Experiment 7 — Loader speed

```bash
python main.py stage=data num_workers=0 batch_size=32  run_name=w1_speed
python main.py stage=data num_workers=0 batch_size=256 run_name=w1_speed
python main.py stage=data num_workers=2 batch_size=128 run_name=w1_speed
python main.py stage=data num_workers=4 batch_size=128 run_name=w1_speed
python main.py stage=data num_workers=4 batch_size=128 augment=strong run_name=w1_speed
```

- Q7.1 Make a table of images/second. Which setting is fastest on your machine?
- Q7.2 Why is `augment=strong` slower?
- Q7.3 More workers is not always faster. Give one reason why.

---



## Part E — Deliverables checklist

- [ ] All `data.py` TODOs and the `TODO [Week 1]` blocks of `main.py` are done; all checks in Part C pass.
- [ ] `python main.py stage=data` prints the summary without errors.
- [ ] Report: outputs + answers for Experiments 1-7, including the four augmentation pictures.
- [ ] Short answers to the `Question for you` comments in `data.py`.

**Next week** you will feed these batches into a ResNet-18 that you build yourself.