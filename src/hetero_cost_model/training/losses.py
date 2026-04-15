"""Loss functions for the cost model."""
import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Margin-based hinge ranking loss over all ordered pairs in a batch.

    Pairs with equal targets are masked out so they do not contribute.
    Returns a scalar that averages over the non-trivial pairs.
    """
    if pred.numel() < 2:
        return pred.new_zeros(())
    diff_pred = pred.unsqueeze(0) - pred.unsqueeze(1)
    diff_true = target.unsqueeze(0) - target.unsqueeze(1)
    sign = torch.sign(diff_true)
    loss = F.relu(margin - sign * diff_pred)
    mask = (sign != 0).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


__all__ = ["pairwise_ranking_loss"]
