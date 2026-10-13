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
        # TODO 1a: create the four layers of the main path:
        #   self.conv1: nn.Conv2d(in_channels -> out_channels, kernel 3, stride=stride, padding=1, bias=False)
        #   self.bn1:   nn.BatchNorm2d(out_channels)
        #   self.conv2: nn.Conv2d(out_channels -> out_channels, kernel 3, stride=1, padding=1, bias=False)
        #   self.bn2:   nn.BatchNorm2d(out_channels)
        #   Why bias=False? (Hint: what does BatchNorm do right after the conv?)

        # TODO 1b: create self.shortcut.
        #   Default: nn.Identity().
        #   If residual AND (stride != 1 or in_channels != out_channels), x and F(x) have
        #   different shapes and cannot be added. Then use a "projection":
        #       nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
        #                     nn.BatchNorm2d(out_channels))
        raise NotImplementedError("TODO 1: BasicBlock.__init__")

    def forward(self, x):
        # TODO 2: implement the diagram above.
        #   out = relu(bn1(conv1(x)))
        #   out = bn2(conv2(out))            <- NO relu here yet!
        #   if self.residual: out = out + shortcut(x)
        #   return relu(out)
        raise NotImplementedError("TODO 2: BasicBlock.forward")


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
        # TODO 3a: the stem (an nn.Sequential stored in self.stem)
        #   "cifar":    Conv2d(3, width, 3, stride 1, padding 1, bias=False) -> BatchNorm2d -> ReLU
        #   "imagenet": Conv2d(3, width, 7, stride 2, padding 3, bias=False) -> BatchNorm2d -> ReLU
        #               -> MaxPool2d(3, stride 2, padding 1)
        #   anything else: raise ValueError(f"Unknown stem: {stem}")

        # TODO 3b: the four stages, built with self._make_layer(in_ch, out_ch, stride, residual):
        #   layer1: width   -> width,   stride 1
        #   layer2: width   -> 2*width, stride 2
        #   layer3: 2*width -> 4*width, stride 2
        #   layer4: 4*width -> 8*width, stride 2

        # TODO 3c: the head
        #   self.pool    = nn.AdaptiveAvgPool2d(1)        (global average pooling)
        #   self.dropout = nn.Dropout(dropout)
        #   self.fc      = nn.Linear(8*width, num_classes)

        # TODO 3d: call self._init_weights()
        raise NotImplementedError("TODO 3: ResNet18.__init__")

    @staticmethod
    def _make_layer(in_channels, out_channels, stride, residual, num_blocks=2):
        # TODO 3e: a stage = num_blocks BasicBlocks in an nn.Sequential.
        #   Only the FIRST block changes channels/stride (in_channels -> out_channels, stride);
        #   the others keep the shape (out_channels -> out_channels, stride 1).
        #   Hint: nn.Sequential(*list_of_blocks)
        raise NotImplementedError("TODO 3e: ResNet18._make_layer")

    def _init_weights(self):
        # TODO 3f: loop over self.modules():
        #   nn.Conv2d      -> nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #   nn.BatchNorm2d -> weight = 1 (nn.init.ones_), bias = 0 (nn.init.zeros_)
        raise NotImplementedError("TODO 3f: ResNet18._init_weights")

    def forward(self, x):
        # TODO 3g: stem -> layer1 -> layer2 -> layer3 -> layer4 -> pool -> flatten(1)
        #   -> dropout -> fc. Return RAW logits (no softmax!). Shapes for the cifar stem:
        #   (B,3,32,32) -> (B,w,32,32) -> (B,w,32,32) -> (B,2w,16,16) -> (B,4w,8,8)
        #   -> (B,8w,4,4) -> pool (B,8w,1,1) -> flatten (B,8w) -> fc (B,num_classes)
        raise NotImplementedError("TODO 3g: ResNet18.forward")


def count_parameters(model):
    """Total number of trainable parameters (standard ResNet-18 on CIFAR: 11,173,962)."""
    # TODO 4: sum p.numel() over model.parameters() that have requires_grad
    raise NotImplementedError("TODO 4: count_parameters")


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
    # TODO 5: one line. Use F.log_softmax(logits, dim=1) (numerically stable!),
    #   NOT torch.log(torch.softmax(...)). Sum over classes (dim=1), mean over the batch.
    raise NotImplementedError("TODO 5: soft_cross_entropy")


def build_optimizer(model, name="sgd", lr=0.1, momentum=0.9, weight_decay=5e-4):
    # TODO 6: return the optimizer for model.parameters():
    #   "sgd"   -> torch.optim.SGD(..., lr, momentum, weight_decay, nesterov=momentum > 0)
    #   "adam"  -> torch.optim.Adam(..., lr, weight_decay)
    #   "adamw" -> torch.optim.AdamW(..., lr, weight_decay)
    #   otherwise raise ValueError(f"Unknown optimizer: {name}")
    raise NotImplementedError("TODO 6: build_optimizer")


def build_scheduler(optimizer, name="cosine", epochs=30):
    # TODO 7: return a learning-rate scheduler from torch.optim.lr_scheduler (stepped once per epoch):
    #   "cosine" -> CosineAnnealingLR(optimizer, T_max=epochs)
    #   "step"   -> StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    #   "linear" -> LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
    #   "none"   -> None
    #   otherwise raise ValueError(f"Unknown scheduler: {name}")
    raise NotImplementedError("TODO 7: build_scheduler")


def gradient_norms(model, images, targets):
    """One forward/backward pass; returns {layer name: L2 norm of its gradients}.

    This shows how strongly each part of the network would be updated. Compare
    residual=True and residual=False in the README experiments.
    """
    # TODO 10:
    #   a) model.train(); model.zero_grad()
    #   b) compute soft_cross_entropy(model(images), targets) and call .backward() on it
    #   c) for name, module in model.named_children():
    #        collect p.grad.norm() ** 2 for every parameter p of module whose p.grad is not None;
    #        if there are any, norms[name] = sqrt(sum of them)   (a Python float, use .item())
    #      (layers without parameters, like pool and dropout, are skipped)
    #   d) model.zero_grad() again (leave no gradients behind) and return norms
    raise NotImplementedError("TODO 10: gradient_norms")


def overfit_single_batch(model, images, targets, optimizer, steps=100, print_every=10):
    """Train repeatedly on ONE batch. A correct model + loss should drive the loss towards 0."""
    model.train()
    losses = []
    for step in range(1, steps + 1):
        # TODO 8: the 5 lines at the heart of ALL deep learning:
        #   1. forward:   logits = model(images)
        #   2. loss:      loss = soft_cross_entropy(logits, targets)
        #   3. reset:     optimizer.zero_grad()     (gradients ACCUMULATE otherwise!)
        #   4. backward:  loss.backward()           (compute d loss / d every parameter)
        #   5. update:    optimizer.step()          (move parameters against the gradient)
        logits, loss = None, None

        losses.append(loss.item())
        if step % print_every == 0 or step == 1:
            acc = (logits.argmax(1) == targets.argmax(1)).float().mean().item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | batch acc {acc:.3f}")
    return losses


def train_one_epoch(model, loader, optimizer, device):
    """One pass over the training data. Returns (mean loss, accuracy)."""
    # TODO 9:
    #   a) model.train()  (turns ON dropout and batch-statistics in BatchNorm)
    #   b) for images, targets in loader:
    #        - move both to `device`
    #        - the same 5 steps as in overfit_single_batch
    #        - accumulate:  total_loss += loss.item() * batch_size
    #                       correct    += number of rows where logits.argmax(1) == targets.argmax(1)
    #                       seen       += batch_size
    #   c) return total_loss / seen, correct / seen
    #   Why multiply the loss by the batch size? (Hint: is the last batch full?)
    raise NotImplementedError("TODO 9: train_one_epoch")
