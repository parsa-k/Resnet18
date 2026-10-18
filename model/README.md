# Week 2 — Building ResNet-18 and the Training Loop (`resnet.py`)

> **Goal of this week:** build a real, 11-million-parameter convolutional neural network
> **from individual layers**, understand *why* its skip connections matter, and write the loop that trains it.
> Before training for hours, you will prove your model is correct with cheap sanity checks.

**Files this week:** copy `resnet.py` into your working folder (next to your Week 1 `main.py` and `data.py`).
Complete all TODOs in `resnet.py` and the `TODO [Week 2]` blocks (5-10) in `main.py`.
**Commands you will run:** `python main.py stage=model ...` and `python main.py stage=train ...`

---

## Part A — Background

### A.1 From pixels to predictions

A neural network is a function $f_\theta(x)$ with millions of adjustable parameters $\theta$.
For CIFAR-10 it maps an image `(3, 32, 32)` to **10 numbers called logits**, one score per class.
Training adjusts $\theta$ so that the logit of the correct class becomes the largest.

### A.2 Convolution

A fully connected layer on a 32×32×3 image would need 3,072 weights *per neuron* and would ignore that
nearby pixels belong together. A **convolution** slides a small filter (e.g. 3×3) over the image and computes
a weighted sum at every position:

* **Local:** each output looks only at a small neighbourhood (its *receptive field*).
* **Weight sharing:** the same filter is used everywhere → an edge detector works in every corner of the image.
* **Many filters:** a layer with `out_channels=64` learns 64 different filters → 64 *feature maps*.

`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)`:

$$\text{out size} = \left\lfloor \frac{\text{in size} + 2\cdot\text{padding} - \text{kernel}}{\text{stride}} \right\rfloor + 1$$

Example: 32×32 input, kernel 3, padding 1, stride 1 → 32×32. With stride 2 → 16×16.

Number of weights: $k \times k \times C_{in} \times C_{out}$ (+ $C_{out}$ biases if `bias=True`).
A 3×3 conv from 64 to 64 channels has $3 \cdot 3 \cdot 64 \cdot 64 = 36{,}864$ weights.

Early layers learn edges and colours; deeper layers combine them into textures, parts (wheels, eyes) and objects.

### A.3 Batch Normalization and ReLU

**BatchNorm** (`nn.BatchNorm2d`) normalizes every channel over the batch to mean 0 / std 1, then applies a learned
scale γ and shift β. It keeps the signal well-scaled through many layers and allows high learning rates.
Because BatchNorm subtracts the mean anyway, a bias in the preceding conv is useless → `bias=False`.

* In **training mode** (`model.train()`) it uses the statistics of the current batch and updates running averages.
* In **evaluation mode** (`model.eval()`) it uses the stored running averages, so a prediction does not depend on the other images in the batch.

**ReLU** $= \max(0, x)$ is the non-linearity. Without it, a stack of linear layers would collapse into one linear layer.

### A.4 Why "residual"? The degradation problem

You would expect a deeper network to be at least as good as a shallow one — it could just copy the shallow one
and set the extra layers to the identity. In practice, He et al. (2015) observed that plain deep networks got
**worse training error** when made deeper: the optimizer struggles to learn even the identity through many
non-linear layers, and the gradient signal gets weaker on its way back to the first layers.

**The residual idea:** let each block learn only a *correction* $F(x)$ to its input:

$$\text{out} = \text{ReLU}\big(F(x) + x\big)$$

* If the identity is optimal, the block just has to learn $F(x) = 0$ — easy (weights near 0).
* During backpropagation the gradient flows through the "+ x" path **unchanged**:
  $\frac{\partial\, \text{out}}{\partial x} = \frac{\partial F}{\partial x} + I$.
  The identity term gives every early layer a direct gradient highway from the loss.

```
x ──► conv3x3(stride) ─► BN ─► ReLU ─► conv3x3 ─► BN ──(+)──► ReLU ──► out
│                                                    ▲
└──────────────── shortcut(x) ───────────────────────┘
```

When a block changes the spatial size (stride 2) or the number of channels, `x` cannot be added directly.
The **projection shortcut** (1×1 conv with the same stride + BatchNorm) reshapes `x` to match.

### A.5 The ResNet-18 architecture (CIFAR version)

| Part | Layers | Output shape (width = 64) | Parameters |
|------|--------|---------------------------|------------|
| input | — | (B, 3, 32, 32) | — |
| stem | 3×3 conv, 64, stride 1 + BN + ReLU | (B, 64, 32, 32) | 1,856 |
| layer1 | 2 × BasicBlock(64), stride 1 | (B, 64, 32, 32) | 147,968 |
| layer2 | 2 × BasicBlock(128), first stride 2 | (B, 128, 16, 16) | 525,568 |
| layer3 | 2 × BasicBlock(256), first stride 2 | (B, 256, 8, 8) | 2,099,712 |
| layer4 | 2 × BasicBlock(512), first stride 2 | (B, 512, 4, 4) | 8,393,728 |
| pool | global average pool | (B, 512, 1, 1) | 0 |
| fc | linear 512 → 10 | (B, 10) | 5,130 |
| **total** | 1 + 8×2 + 1 = **18 weighted layers** | | **11,173,962** |

Each stage halves the resolution and doubles the channels: the network trades *where* for *what*.

**Stem:** the original ImageNet ResNet starts with a 7×7 stride-2 conv + max-pool, shrinking 224×224 images to 56×56.
On 32×32 CIFAR images, that would leave only 8×8 after the stem — too much information is thrown away immediately.
`stem="cifar"` uses a 3×3 stride-1 conv instead. You can compare both.

**Global average pooling** averages each 4×4 feature map to one number → 512 features, regardless of image size,
with no parameters. **Dropout** (before `fc`) randomly zeros features during training — you will study it in Week 3.

**Weight initialization:** Kaiming (He) initialization draws conv weights with variance $2/\text{fan}$, chosen so that
the signal keeps roughly the same scale through ReLU layers at the start of training.

### A.6 From logits to a loss: softmax cross-entropy

Softmax turns logits $z$ into probabilities: $p_c = e^{z_c} / \sum_j e^{z_j}$.
The **cross-entropy** between the target distribution $y$ (your one-hot / smoothed vector from Week 1) and $p$:

$$\mathcal{L} = -\sum_{c} y_c \log p_c$$

For a hard one-hot target this is just $-\log p_{\text{true class}}$: confident and right → loss ≈ 0; confident and wrong → huge loss.
We compute $\log p$ with `F.log_softmax`, which is numerically stable (`log(softmax(z))` can produce `log(0) = -inf`).

**Sanity value:** a freshly initialized network outputs ≈ uniform probabilities $p_c \approx 1/10$,
so the initial loss should be close to $-\log(1/10) = \ln 10 \approx 2.303$.

### A.7 The training loop

Every deep-learning training loop is the same five steps per batch:

```python
logits = model(images)                    # 1. forward pass
loss = soft_cross_entropy(logits, targets)  # 2. how wrong are we?
optimizer.zero_grad()                     # 3. clear old gradients (they accumulate!)
loss.backward()                           # 4. backpropagation: d loss / d every parameter
optimizer.step()                          # 5. update the parameters
```

One pass over the whole training set is an **epoch**.

### A.8 Optimizers, weight decay and learning-rate schedules

* **SGD:** $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$, where η is the **learning rate**: too small → slow; too large → divergence.
* **Momentum (0.9):** keeps a running average of past gradients, like a ball rolling downhill, which smooths noisy steps. *Nesterov* is a slightly improved version.
* **Adam:** adapts a separate step size for every parameter using running gradient statistics; works out of the box with lr ≈ 1e-3.
* **AdamW:** Adam with *decoupled* weight decay (the "correct" way to combine Adam and weight decay).
* **Weight decay:** adds a penalty $\lambda \lVert\theta\rVert^2$ that keeps weights small, which reduces overfitting (Week 3).
* **LR schedule:** start with a large learning rate to make fast progress, then lower it to fine-tune:
  `cosine` (smooth decay to 0), `step` (÷10 every third of training), `linear`, or `none`.

### A.9 Sanity checks before you train for hours

1. **Shapes:** print the output shape of every stage; compare with the table above.
2. **Parameter count:** 11,173,962 for the standard model.
3. **Initial loss ≈ ln 10.** Much larger → wrong initialization or loss; NaN → bug.
4. **Your loss = PyTorch's loss** (`F.cross_entropy` accepts probability targets).
5. **Overfit one batch:** a correct model must be able to *memorize* 128 images, with the loss going to ~0 and the batch accuracy to 100 %.
   If it can't, there is a bug (wrong loss, forgot `zero_grad`, wrong labels, learning rate far off...).
   This is the single most useful debugging trick in deep learning.

---

## Part B — Your tasks

**`resnet.py`** (in this order)

1. `BasicBlock.__init__` — conv/BN layers and the shortcut (identity or projection)
2. `BasicBlock.forward` — with and without the residual connection
3. `ResNet18` — stem, `_make_layer`, head, `_init_weights`, `forward`
4. `count_parameters`
5. `soft_cross_entropy`
6. `build_optimizer` — sgd / adam / adamw
7. `build_scheduler` — cosine / step / linear / none
8. `overfit_single_batch` — the five training steps
9. `train_one_epoch` — the same steps over the whole loader + loss/accuracy bookkeeping
10. `gradient_norms`

**`main.py`** — `TODO [Week 2]` blocks: 6 `build_model`, 7 `run_model_stage`, 8 `train` (without the Week 3 part), 10 `run_train_stage`.

---

## Part C — Step-by-step checks

```bash
# 1-2. one block: shapes and the projection shortcut
python -c "
import torch, resnet
b = resnet.BasicBlock(64, 128, stride=2)
print(b(torch.randn(2, 64, 32, 32)).shape, type(b.shortcut).__name__)
print(type(resnet.BasicBlock(64, 64).shortcut).__name__, type(resnet.BasicBlock(64, 128, 2, residual=False).shortcut).__name__)"
# torch.Size([2, 128, 16, 16]) Sequential
# Identity Identity

# 3-4. whole network
python -c "
import torch, resnet
m = resnet.ResNet18()
print(m(torch.randn(2, 3, 32, 32)).shape, f'{resnet.count_parameters(m):,}')"
# torch.Size([2, 10]) 11,173,962

# 5. loss: logits [2, 0, 0] with target class 0 -> -log(e^2 / (e^2 + 2)) = 0.2395
python -c "
import torch, resnet
print(resnet.soft_cross_entropy(torch.tensor([[2.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])).item())"
# 0.2395448386669159

# 6-7.
python -c "
import resnet
m = resnet.ResNet18(width=8)
for o in ('sgd', 'adam', 'adamw'): print(type(resnet.build_optimizer(m, o)).__name__)
opt = resnet.build_optimizer(m)
for s in ('cosine', 'step', 'linear', 'none'): print(s, type(resnet.build_scheduler(opt, s, 30)).__name__)"
# SGD Adam AdamW / cosine CosineAnnealingLR, step StepLR, linear LinearLR, none NoneType
```

Then the full model stage:

```bash
python main.py stage=model
```

Expected output (exact numbers vary slightly with hardware):

```
ResNet-18 (residual, stem=cifar, width=64): 11,173,962 parameters
input    -> (2, 3, 32, 32)
stem     -> (2, 64, 32, 32)             1,856 params
layer1   -> (2, 64, 32, 32)           147,968 params
layer2   -> (2, 128, 16, 16)          525,568 params
layer3   -> (2, 256, 8, 8)          2,099,712 params
layer4   -> (2, 512, 4, 4)          8,393,728 params
pool     -> (2, 512, 1, 1)                  0 params
dropout  -> (2, 512)                        0 params
fc       -> (2, 10)                     5,130 params

Loss at initialization: 2.4... (PyTorch: 2.4..., ln(10) = 2.3026)

Gradient norm per layer (one batch):
  stem   ...
Overfitting a single batch of 128 images for 100 steps:
  step    1 | loss 2.4... | batch acc 0.1..
  step   10 | ...
  step   20 | loss 0.01.. | batch acc 1.000
  ...
Loss went from 2.4... to 0.000...
```

On a slow CPU, use a smaller model for the experiments below: add `width=32` or `width=16`.

---

## Part D — Experiments: change the model, watch the output

Write the outputs (or the important lines) and your answers into your report.

### Experiment 1 — Shapes and the stem

```bash
python main.py stage=model overfit_steps=1 stem=cifar
python main.py stage=model overfit_steps=1 stem=imagenet
```

* Q1.1 Write down the spatial size after each stage for both stems. Verify **one** of them by hand with the conv output-size formula.
* Q1.2 With `stem=imagenet`, `layer4` outputs 1×1 feature maps. What is a 3×3 convolution doing on a 1×1 input (with padding 1)?
* Q1.3 Predict: which stem will reach higher accuracy on CIFAR-10? (Test it in Week 3.)

### Experiment 2 — Model size

```bash
python main.py stage=model overfit_steps=1 width=16
python main.py stage=model overfit_steps=1 width=32
python main.py stage=model overfit_steps=1 width=64
```

* Q2.1 Table: width → total parameters. When width doubles, the parameter count grows by a factor of about ___. Explain with the formula $k^2 C_{in} C_{out}$.
* Q2.2 Which stage holds most of the parameters? Why?
* Q2.3 Compute by hand the parameters of `layer1` for width 64 (two blocks, each with 2 convs of 3×3×64×64 and 2 BatchNorms with 2×64 parameters each). Does it match 147,968?

### Experiment 3 — Initial loss and the loss function

```bash
python main.py stage=model overfit_steps=1 seed=1
python main.py stage=model overfit_steps=1 seed=2
python main.py stage=model overfit_steps=1 label_smoothing=0.1
```

* Q3.1 Is the initial loss close to ln 10? Why is it not exactly 2.3026?
* Q3.2 Do your loss and PyTorch's loss always agree, including with label smoothing?
* Q3.3 *Break it on purpose:* temporarily replace `log_softmax` by `torch.log(torch.softmax(...))` and
  multiply the logits by 1000 inside `soft_cross_entropy`. What happens? Undo the change afterwards.

### Experiment 4 — Overfit a single batch (the most important sanity check)

```bash
python main.py stage=model width=32 lr=0.1                      # baseline
python main.py stage=model width=32 lr=0.01
python main.py stage=model width=32 lr=1.0
python main.py stage=model width=32 optimizer=adam lr=0.001
python main.py stage=model width=32 momentum=0.0
```

* Q4.1 For each setting: after how many steps does the batch accuracy reach 1.000? What is the final loss?
* Q4.2 What happens with `lr=1.0`? Explain using the idea of gradient descent step size.
* Q4.3 *Break it on purpose:* comment out `optimizer.zero_grad()` in `overfit_single_batch` and run again. What changes and why? Undo the change.
* Q4.4 Memorizing 128 images perfectly is **not** the same as learning to classify. Why is this still a useful test?

### Experiment 5 — Residual vs. plain network

```bash
python main.py stage=model width=32 residual=True  overfit_steps=60
python main.py stage=model width=32 residual=False overfit_steps=60
python main.py stage=train width=16 subset_fraction=0.1 epochs=5 residual=True  run_name=w2_res
python main.py stage=train width=16 subset_fraction=0.1 epochs=5 residual=False run_name=w2_plain
```

* Q5.1 Compare the gradient norms per layer and the speed of the single-batch overfit. Which network memorizes faster?
* Q5.2 Compare the training loss after 5 epochs (`runs/w2_res/history.json` vs `runs/w2_plain/history.json`).
* Q5.3 BatchNorm already keeps the signal well-scaled, and 18 layers is not very deep. Still, skip connections usually help
  clearly (you will measure it over 20 epochs in Week 3, Lab G). The paper saw the degradation problem most strongly at 34+ layers.
  Explain why BatchNorm reduces the problem but does not solve it, and why depth makes it worse.

### Experiment 6 — Your first real training run

```bash
python main.py stage=train width=16 subset_fraction=0.1 epochs=10 scheduler=cosine run_name=w2_cos
python main.py stage=train width=16 subset_fraction=0.1 epochs=10 scheduler=none   run_name=w2_none
python main.py stage=train width=16 subset_fraction=0.1 epochs=10 scheduler=step   run_name=w2_step
```

* Q6.1 Plot (or tabulate) training loss and the `lr` column per epoch for the three schedules. (Tip: `history.json` contains both;
  a few lines of matplotlib are enough — or wait for `plot_history` next week.)
* Q6.2 Which schedule reaches the lowest training loss? Do you see a jump when `step` lowers the learning rate?
* Q6.3 The training accuracy keeps going up. Does that mean the model is getting better at classifying *new* images?
  What would you need to know that? (This is next week's topic.)

### Experiment 7 — Speed (optional)

Run `stage=train epochs=1 subset_fraction=0.1` with `width=16`, `32` and `64` and note the seconds per epoch.
If you have a GPU, compare `cuda`/`mps` with the CPU (temporarily return `torch.device("cpu")` in `get_device`).

---

## Part E — Deliverables checklist

- [ ] All `resnet.py` TODOs and the `TODO [Week 2]` blocks of `main.py` are done; Part C checks pass.
- [ ] `python main.py stage=model` shows 11,173,962 parameters, initial loss ≈ 2.3-2.5 and overfits the batch.
- [ ] `python main.py stage=train ...` prints a decreasing training loss.
- [ ] Report: outputs + answers for Experiments 1-6.

**Next week** you will add validation, a proper evaluation, and run a full set of experiments on
overfitting, data leakage, augmentation, dropout and hyperparameter tuning.
