import torch.nn
from torch.nn.functional import cross_entropy


def reed_soft(logits, targets, beta=0.95, reduction='none'):
    """
    Soft version of Reed et al 2014.
    Equivalent to entropy regularization.
    """
    assert reduction == 'none'  # stupid quick hack
    cross_entropy_loss = cross_entropy(logits, targets, reduction=reduction)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs.log() * probs).sum(dim=-1)
    loss = cross_entropy_loss * beta - entropy * (1 - beta)
    return loss


def reed_hard(logits, targets, beta=0.8, reduction='none'):
    """
    Soft version of Reed et al 2014.
    Equivalent to entropy regularization.
    """
    cross_entropy_loss = cross_entropy(logits, targets, reduction=reduction)
    most_confident_probs = torch.softmax(logits, dim=-1).max(dim=-1)[0]
    loss = cross_entropy_loss * beta - (1 - beta) * most_confident_probs.log()
    return loss


losses = {
    "cross-entropy": cross_entropy,
    "reed-soft": reed_soft,
    "reed-hard": reed_hard,
}

__all__ = ["losses"]
