"""Auxiliary losses over a ViT patch grid (REVERSE_LEARNING.md): CTC (2A), text-to-visual
reconstruction (3) and cross-attention entropy (4). Training-time only; inference is unchanged."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CtcHead(nn.Module):
    """Mean-pool patch tokens over height -> width frames -> Linear to vocab + blank."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.norm, self.proj = nn.LayerNorm(dim), nn.Linear(dim, vocab_size + 1)
        self.blank = vocab_size

    def forward(self, hidden: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        return self.proj(self.norm(pool_frames(hidden, grid_h, grid_w)))  # (B, W, V+1)


def pool_frames(hidden: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """(B, [CLS +] H*W, d) patch tokens -> (B, W, d) frames, mean-pooled over height."""
    patches = hidden[:, -grid_h * grid_w:]  # drop a leading CLS token if present
    return patches.reshape(hidden.shape[0], grid_h, grid_w, -1).mean(dim=1)


def ctc_loss(head: CtcHead, hidden, labels, grid_h, grid_w, special_ids) -> torch.Tensor:
    """labels: (B, L) token ids, -100 = pad. Special tokens are dropped from the targets."""
    logits = head(hidden, grid_h, grid_w).float()
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, C)
    if log_probs.device.type == "mps":  # aten::_ctc_loss has no MPS kernel
        log_probs, labels = log_probs.cpu(), labels.cpu()
    keep = labels != -100
    for s in special_ids:
        keep &= labels != s
    targets = [row[mask] for row, mask in zip(labels, keep)]
    lengths = torch.tensor([len(t) for t in targets])
    inputs = torch.full((labels.shape[0],), logits.shape[1], dtype=torch.long)
    # ponytail: infeasible rows (target > frames) are zeroed by zero_infinity, not filtered
    return F.ctc_loss(log_probs, torch.cat(targets), inputs, lengths, blank=head.blank,
                      zero_infinity=True, reduction="mean")


def infeasible_fraction(label_lengths, frames: int) -> float:
    """Share of rows CTC cannot fit (target length > frames)."""
    return sum(n > frames for n in label_lengths) / max(len(label_lengths), 1)


class ReconHead(nn.Module):
    """Reverse decoder: learned frame queries cross-attend to text embeddings -> visual frames."""

    def __init__(self, dim: int, frames: int, layers: int = 2, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(frames, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(dim, heads, dim_feedforward=2 * dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.proj = nn.Linear(dim, dim)

    def forward(self, text: torch.Tensor, text_pad: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(text.shape[0], -1, -1)
        return self.proj(self.decoder(queries, text, memory_key_padding_mask=text_pad))


def recon_loss(head: ReconHead, embed: nn.Module, hidden, labels, grid_h, grid_w, pad_id) -> torch.Tensor:
    """MSE between text-predicted frames and the (normalised, NOT detached) encoder frames."""
    text_pad = labels == -100
    text = embed(labels.masked_fill(text_pad, pad_id))
    target = pool_frames(hidden, grid_h, grid_w).float()
    target = F.layer_norm(target, target.shape[-1:])
    return F.mse_loss(head(text, text_pad).float(), target)


def alignment_entropy(cross_attn: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean entropy of one layer's cross-attention (B, H, U, S), head-averaged, over real tokens."""
    p = cross_attn.float().mean(dim=1)  # (B, U, S)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)  # (B, U)
    keep = labels != -100
    return entropy[keep].mean()
