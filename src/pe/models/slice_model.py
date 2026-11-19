"""The slice model: a timm backbone, dropout and one output, pe_present_on_image (04_slice_models)."""
import timm
import torch
from torch import nn

# The 8 candidates of 04, with the weights the runs use
BACKBONES = {"seresnext50": "seresnext50_32x4d.racm_in1k", "seresnext101": "seresnext101_32x4d.gluon_in1k", "resnest50": "resnest50d.in1k",
             "efficientnet_b0": "tf_efficientnet_b0.ns_jft_in1k", "efficientnet_b4": "tf_efficientnet_b4.ns_jft_in1k",
             "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k", "mobilevitv2_175": "mobilevitv2_175.cvnets_in22k_ft_in1k", "rexnet_200": "rexnet_200.nav_in1k"}


class SliceModel(nn.Module):
    """Input: windowed slices in [0, 1], one channel per slice. Output: `(logits, features)`.

    The channels are not colours, so every channel gets the mean of the three means and stds the weights were trained with.
    """

    def __init__(self, backbone: str, in_chans: int = 3, pretrained: bool = True, dropout: float = 0.2, series_targets: int = 0):
        super().__init__()
        self.net = timm.create_model(BACKBONES[backbone], pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.net.num_features, 1)
        if series_targets:
            self.series_head = nn.Linear(self.net.num_features, series_targets)  # used only in training, on the features
        config = self.net.pretrained_cfg
        self.register_buffer("mean", torch.tensor(sum(config["mean"]) / len(config["mean"])), persistent=False)
        self.register_buffer("std", torch.tensor(sum(config["std"]) / len(config["std"])), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net((x - self.mean) / self.std)
        with torch.autocast(x.device.type, enabled=False):  # the head in float32, so the logits are not rounded to bfloat16
            return self.head(self.dropout(features.float())).squeeze(1), features
