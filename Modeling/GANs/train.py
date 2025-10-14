import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Modeling.GANs.gan_wrapper import GAN_BG_Macro
from Modeling.GANs.data_loader import MacroGANData
from Optimizers.DSP import JointReturnSimulator
import matplotlib.pyplot as plt
import os
import numpy as np


def train_gan(data_object, device='cpu', epochs=100, batch_size=64, latent_dim=30, ell=9, window=30, num_macro=4, save_dir=None):
    dataset = MacroGANData(data_object, ell=ell, window=window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GAN_BG_Macro(latent_dim=latent_dim, window=window, ell=ell, num_macro=num_macro).to(device)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    loss_g_list, loss_d_list = [], []
    generated_samples = []

    for epoch in range(epochs):
        for past_returns, macro, real_returns, date in loader:
            past_returns, macro, real_returns = past_returns.to(device), macro.to(device), real_returns.to(device)

            z = torch.randn(batch_size, ell, latent_dim).to(device)


            fake_returns = model.generate(z, past_returns, macro)

            # Train Discriminator
            opt_d.zero_grad()
            logits_real = model.discriminate(real_returns, past_returns, macro)
            logits_fake = model.discriminate(fake_returns.detach(), past_returns, macro)
            loss_d = loss_fn(logits_real, torch.ones_like(logits_real)) + \
                     loss_fn(logits_fake, torch.zeros_like(logits_fake))
            loss_d.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            logits_fake = model.discriminate(fake_returns, past_returns, macro)
            loss_g = loss_fn(logits_fake, torch.ones_like(logits_fake))
            loss_g.backward()
            opt_g.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")
        loss_g_list.append(loss_g.item())
        loss_d_list.append(loss_d.item())
        generated_samples.append(fake_returns.detach().cpu().numpy())

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))

    # Plot losses
    plt.figure(figsize=(8,4))
    plt.plot(loss_d_list, label='Discriminator')
    plt.plot(loss_g_list, label='Generator')
    plt.title('GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot sample return paths vs real
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    for i in range(3):
        axs[i].plot(fake_returns[i].detach().cpu().numpy(), label='Generated')
        axs[i].plot(real_returns[i].detach().cpu().numpy(), label='Real', alpha=0.5)
        axs[i].legend()
        axs[i].set_title(f"Sample {i+1}: Generated vs. Real")
    plt.tight_layout()
    plt.show()

    # Evaluate autocorrelation of generated returns
    gen_flat = np.concatenate(generated_samples).flatten()
    acorr = np.correlate(gen_flat - np.mean(gen_flat), gen_flat - np.mean(gen_flat), mode='full')
    acorr = acorr[acorr.size // 2:]
    acorr /= acorr[0]
    plt.figure(figsize=(6,3))
    plt.plot(acorr[:20])
    plt.title("Autocorrelation of Generated Returns")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model

