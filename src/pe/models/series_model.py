"""The series model: a bidirectional GRU over the slices of a series, one output per slice and nine per series (05_series_models).

1st place's recipe: the GRU reads every slice's features in body order; a linear head on each step predicts
pe_present_on_image, and the series labels come from the steps pooled two ways, by their maximum and by attention.
Padding is masked everywhere, so a series gives the same outputs alone or in a padded batch.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeriesModel(nn.Module):
    """Input: features (batch, slices, n_in) and the number of real slices of each series. Output: `(slice logits, series logits)`."""

    def __init__(self, n_in: int, hidden: int = 256, n_series: int = 9, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True, bidirectional=True)
        self.slice_head = nn.Linear(2 * hidden, 1)
        self.attention = nn.Linear(2 * hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.series_head = nn.Linear(4 * hidden, n_series)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=x.shape[1])
        mask = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
        slice_logits = self.slice_head(h).squeeze(-1)
        most = h.masked_fill(~mask[..., None], float("-inf")).amax(1)
        weights = self.attention(h).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(1)
        pooled = torch.cat([most, (weights[..., None] * h).sum(1)], dim=1)
        return slice_logits, self.series_head(self.dropout(pooled))
