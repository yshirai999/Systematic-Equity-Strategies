import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from encoder import SelfAttention

class Generator(nn.Module):
    def __init__(self, latent_dim, ell, window, cond_dim=64, kernel_size=3, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.ell = ell
        self.window = window
        self.embed_dim = embed_dim

        self.z_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(latent_dim * ell, embed_dim),
            nn.ReLU()
        )

        self.cond_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(window * cond_dim, embed_dim),
            nn.ReLU()
        )

        self.joint_proj = nn.Linear(2 * embed_dim, latent_dim * embed_dim)

        self.resblock1 = nn.ModuleList([
            nn.ModuleList([
                spectral_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2)),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                spectral_norm(nn.Conv1d(hidden_dim, embed_dim, 1)),
            ]) for _ in range(2)
        ])

        self.attn1 = SelfAttention(embed_dim, num_heads=4)

        self.resblock2 = nn.ModuleList([
            nn.ModuleList([
                spectral_norm(nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2)),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                spectral_norm(nn.Conv1d(hidden_dim, embed_dim, 1)),
            ]) for _ in range(2)
        ])

        self.attn2 = SelfAttention(embed_dim, num_heads=4)

        self.final = spectral_norm(nn.Conv1d(embed_dim, ell, 1))

    def forward(self, z, cond):  # z: [B, latent_dim, ell], cond: [B, window, cond_dim]
        z_embed = self.z_encoder(z)                        # [B, embed_dim]
        cond_embed = self.cond_encoder(cond)               # [B, embed_dim]
        joint = torch.cat([z_embed, cond_embed], dim=1)    # [B, 2 * embed_dim]
        x = self.joint_proj(joint).view(-1, self.latent_dim, self.embed_dim)  # [B, latent_dim, embed_dim]
        x = x.transpose(1, 2)                               # [B, embed_dim, latent_dim]

        for block in self.resblock1:
            residual = x
            x = block[0](x)
            x = block[1](x.transpose(1, 2)).transpose(1, 2)
            x = block[2](x)
            x = block[3](x)
            x += residual

        x = x.transpose(1, 2)
        x = self.attn1(x)
        x = x.transpose(1, 2)

        for block in self.resblock2:
            residual = x
            x = block[0](x)
            x = block[1](x.transpose(1, 2)).transpose(1, 2)
            x = block[2](x)
            x = block[3](x)
            x += residual

        x = x.transpose(1, 2)
        x = self.attn2(x)
        x = x.transpose(1, 2)

        out = self.final(x)                                # [B, ell, latent_dim]
        return out.transpose(1, 2)                          # [B, latent_dim, ell]



