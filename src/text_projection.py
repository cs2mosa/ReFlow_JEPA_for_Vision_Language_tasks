"""
Text projection head g_T: 2-layer MLP into the shared calibrated space, mirroring
Q-Pool's g_v. Used for both the online (trainable) and target (EMA) text pipelines --
same architecture, different weight-update rule (see encoders.make_ema_copy / ema_update).
"""
import torch.nn as nn


class TextProjectionHead(nn.Module):
    def __init__(self, d_text: int, d_shared: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_text, d_shared), nn.LayerNorm(d_shared), nn.GELU(),
            nn.Linear(d_shared, d_shared),
        )

    def forward(self, z_t_raw):
        return self.net(z_t_raw)
