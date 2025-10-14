import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import torch
from Modeling.GANs.train import train_gan
from Data.DataProcessing import data

if __name__ == '__main__':
    d = data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    trained_gan = train_gan(d, device=device, epochs=1, batch_size=8, save_dir="Modeling/GANs/checkpoints")
    print("✅ Forward pass completed.")

