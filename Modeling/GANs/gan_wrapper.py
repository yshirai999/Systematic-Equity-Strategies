import torch
import torch.nn as nn
from encoder import ConditioningEncoder
from generator import Generator
from discriminator import Discriminator

class GAN_BG_Macro(nn.Module):
    def __init__(self, latent_dim, window, ell, num_assets, num_macro):
        super().__init__()
        self.latent_dim = latent_dim
        self.ell = ell
        self.num_assets = num_assets
        self.num_macro = num_macro
        self.window = window

        self.generator = Generator(
            latent_dim=latent_dim, 
            ell=ell, 
            cond_dim=0,  # no longer needed if generator no longer uses conditioning
            embed_dim=32,
            hidden_dim=64
        )

        self.discriminator = Discriminator(
            ell=ell, 
            num_assets=num_assets, 
            macro_dim=num_macro
        )

    def generate(self, z):
        return self.generator(z)  # z: [batch, latent_dim, num_assets]

    def discriminate(self, returns, past_returns, macro):
        return self.discriminator(returns, past_returns, macro)

