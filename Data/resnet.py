"""Week 2 - ResNet-18 implemented from scratch, plus the training step.

Architecture (He et al., 2015, "Deep Residual Learning for Image Recognition"):

    stem  -> conv1 (+ optional maxpool)
    layer1: 2 x BasicBlock,  64 channels, stride 1
    layer2: 2 x BasicBlock, 128 channels, stride 2
    layer3: 2 x BasicBlock, 256 channels, stride 2
    layer4: 2 x BasicBlock, 512 channels, stride 2
    head  -> global average pool -> (dropout) -> fully connected

    1 stem conv + 8 blocks * 2 convs + 1 fc = 18 weighted layers.

Complete every block marked TODO. Read README.md of Week 2 first.
Suggested order (test after each group, see README "Step-by-step checks"):
    1-2. BasicBlock            5. soft_cross_entropy       8.  overfit_single_batch
    3.   ResNet18              6. build_optimizer          9.  train_one_epoch
    4.   count_parameters      7. build_scheduler          10. gradient_norms
Also finish the "TODO [Week 2]" blocks in main.py.

Useful documentation:
    torch.nn layers: https://pytorch.org/docs/stable/nn.html
    optimizers:      https://pytorch.org/docs/stable/optim.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 convs with a skip connection: out = relu(F(x) + shortcut(x)).

    residual=False removes the skip connection (a "plain" block: out = relu(F(x))).

        x ──► conv3x3(stride) ─► BN ─► ReLU ─► conv3x3(1) ─► BN ──(+)──► ReLU ──► out
        │                                                         ▲
        └──────────────────── shortcut(x) ────────────────────────┘
    """

    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if residual and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual:
            out = out + self.shortcut(shortcut)
        return F.relu(out)


class ResNet18(nn.Module):
    """
    stem="cifar":    3x3 conv, stride 1, no maxpool. Keeps 32x32 resolution; best for CIFAR.
    stem="imagenet": 7x7 conv, stride 2, + maxpool. The original design for 224x224 images;
                     on 32x32 inputs it shrinks feature maps to 8x8 right away.
    width:           channels of layer1 (64 in the paper); later layers use 2x, 4x, 8x.
    dropout:         dropout probability before the final classifier.
    residual:        False turns every BasicBlock into a plain block (no skip connections).
    """

    def __init__(self, num_classes=10, stem="cifar", width=64, dropout=0.0, residual=True):
        super().__init__()
        if stem == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
        elif stem == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError(f"Unknown stem: {stem}")

        self.layer1 = self._make_layer(width, width, stride=1, residual=residual)
        self.layer2 = self._make_layer(width, 2 * width, stride=2, residual=residual)
        self.layer3 = self._make_layer(2 * width, 4 * width, stride=2, residual=residual)
        self.layer4 = self._make_layer(4 * width, 8 * width, stride=2, residual=residual)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8 * width, num_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(in_channels, out_channels, stride, residual, num_blocks=2):
        blocks = [BasicBlock(in_channels, out_channels, stride=stride, residual=residual)]
        blocks.extend(BasicBlock(out_channels, out_channels, stride=1, residual=residual) for _ in range(1, num_blocks))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def count_parameters(model):
    """Total number of trainable parameters (standard ResNet-18 on CIFAR: 11,173,962)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trace_shapes(model, input_size=(2, 3, 32, 32)):
    """Print the output shape and parameter count of every top-level part of the model."""
    device = next(model.parameters()).device
    shapes = []
    hooks = [module.register_forward_hook(
                 lambda mod, inp, out, name=name: shapes.append((name, tuple(out.shape))))
             for name, module in model.named_children()]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size, device=device))
    model.train(was_training)
    for hook in hooks:
        hook.remove()

    modules = dict(model.named_children())
    print(f"{'input':8s} -> {str(tuple(input_size)):20s}")
    for name, shape in shapes:
        print(f"{name:8s} -> {str(shape):20s} {count_parameters(modules[name]):>12,} params")


def soft_cross_entropy(logits, targets):
    """Cross-entropy for one-hot (or label-smoothed) target vectors.

    logits:  (B, C) raw scores from the model
    targets: (B, C) probability vectors (each row sums to 1)
    loss = mean over the batch of  -sum_c targets[c] * log softmax(logits)[c]
    """
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def build_optimizer(model, name="sgd", lr=0.1, momentum=0.9, weight_decay=5e-4):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=momentum > 0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, name="cosine", epochs=30):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def gradient_norms(model, images, targets):
    """One forward/backward pass; returns {layer name: L2 norm of its gradients}.

    This shows how strongly each part of the network would be updated. Compare
    residual=True and residual=False in the README experiments.
    """
    model.train()
    model.zero_grad()
    loss = soft_cross_entropy(model(images), targets)
    loss.backward()

    norms = {}
    for name, module in model.named_children():
        total = 0.0
        for param in module.parameters(recurse=True):
            if param.grad is not None:
                total += param.grad.norm().item() ** 2
        if total > 0.0:
            norms[name] = total ** 0.5

    model.zero_grad()
    return norms


def overfit_single_batch(model, images, targets, optimizer, steps=100, print_every=10):
    """Train repeatedly on ONE batch. A correct model + loss should drive the loss towards 0."""
    model.train()
    losses = []
    for step in range(1, steps + 1):
        logits = model(images)
        loss = soft_cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % print_every == 0 or step == 1:
            acc = (logits.argmax(1) == targets.argmax(1)).float().mean().item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | batch acc {acc:.3f}")
    return losses


def train_one_epoch(model, loader, optimizer, device):
    """One pass over the training data. Returns (mean loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = soft_cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == targets.argmax(1)).sum().item()
        seen += batch_size

    return total_loss / seen, correct / seen
