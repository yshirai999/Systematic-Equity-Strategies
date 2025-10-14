import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)

class ConditioningEncoder(nn.Module):
    def __init__(self, num_assets, window, num_macro, cond_dim=64, hidden_cnn=64, hidden_macro=32):
        super().__init__()
        self.cond_dim = cond_dim
        self.seq_len = window

        self.cnn = nn.Sequential(
            nn.Conv1d(num_assets, hidden_cnn, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_cnn, hidden_cnn, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1)  # shape: [B, hidden_cnn, 1]
        )

        self.mlp_macro = nn.Sequential(
            nn.Linear(num_macro, hidden_macro),
            nn.LayerNorm(hidden_macro),
            nn.LeakyReLU()
        )

        # Final input to linear: flattened [hidden_cnn * 1 + hidden_macro]
        self.output_layer = nn.Linear(hidden_cnn + hidden_macro, cond_dim)

    def forward(self, past_returns, macro):
        x = self.cnn(past_returns).squeeze(-1)  # [B, hidden_cnn]
        m = self.mlp_macro(macro)               # [B, hidden_macro]
        out = torch.cat([x, m], dim=1)          # [B, hidden_cnn + hidden_macro]
        return self.output_layer(out)           # [B, cond_dim]



