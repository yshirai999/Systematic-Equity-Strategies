import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, ell, num_assets, macro_dim, hidden_dim=64):
        super().__init__()

        self.ell = ell
        self.num_assets = num_assets
        self.macro_dim = macro_dim

        # === Conditioning embeddings ===
        self.cond_proj_returns = nn.Linear(ell * num_assets, ell)   # flatten past_returns
        self.cond_proj_macro = nn.Linear(macro_dim, ell)            # macro directly → [batch, ell]

        # === Conv layers ===
        self.conv1 = spectral_norm(nn.Conv1d(num_assets + 2, 64, kernel_size=3, padding=1))  # 2 extra channels from cond
        self.norm1 = nn.LayerNorm(ell)
        self.act1 = nn.LeakyReLU(0.2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = spectral_norm(nn.Conv1d(64, 128, kernel_size=3, padding=1))
        self.norm2 = nn.LayerNorm(ell)
        self.act2 = nn.LeakyReLU(0.2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = spectral_norm(nn.Linear(128, 1))

    def forward(self, returns, past_returns, macro):
        # Input: 
        # - returns: [batch, ell, num_assets]
        # - past_returns: [batch, ell, num_assets]
        # - macro: [batch, macro_dim]

        b = returns.size(0)

        # === Project condition inputs to shape [batch, ell] ===
        past_cond = self.cond_proj_returns(past_returns.view(b, -1))  # flatten and project
        macro_cond = self.cond_proj_macro(macro)                      # direct projection
        cond = torch.stack([past_cond, macro_cond], dim=1)            # shape: [batch, 2, ell]

        # === Prepare main input ===
        x = returns.transpose(1, 2)                                   # [batch, num_assets, ell]
        x = torch.cat([x, cond], dim=1)                               # [batch, num_assets+2, ell]

        # === Conv blocks ===
        x = self.conv1(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.act2(x)

        # === Output ===
        x = self.pool(x).squeeze(-1)
        return self.out(x)

