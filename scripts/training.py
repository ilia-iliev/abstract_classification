import math

import numpy as np
import torch

from classifier.modeling import weighted_multilabel_loss


def train_one_epoch(model, data, optimizer, scheduler, device, gradient_accumulation, label_count):
    """Train one epoch with mean-loss gradient accumulation."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batches = len(data)
    for batch_index, (tokens, targets) in enumerate(data, 1):
        logits = model(**{key: value.to(device) for key, value in tokens.items()})
        loss = weighted_multilabel_loss(
            logits, targets.to(device), np.ones(label_count, dtype=np.float32)
        )
        total_loss += loss.item()
        remainder = batches % gradient_accumulation
        group_size = (
            remainder
            if remainder and batch_index > batches - remainder
            else gradient_accumulation
        )
        (loss / group_size).backward()
        if batch_index % gradient_accumulation == 0 or batch_index == batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return {
        "epoch": 1,
        "mean_training_loss": total_loss / batches,
        "optimizer_steps": math.ceil(batches / gradient_accumulation),
    }
