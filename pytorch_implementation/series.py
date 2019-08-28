import numpy as np
import os
from models import vae
import glob
import torch
from hyperparameters import *

DATA_DIR = "data"
SERIES_DIR = "series"
model_path_name = "checkpoints/vae_checkpoint.pth"

if __name__ == "__main__":
    if not os.path.exists(SERIES_DIR):
        os.makedirs(SERIES_DIR)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vae.ConvVAE(VAE_Z_SIZE, VAE_KL_TOLERANCE)
    if os.path.exists("checkpoints/vae_checkpoint.pth"):
        model.load_state_dict(torch.load("checkpoints/vae_checkpoint.pth"))
    model.to(device)
    model.eval()

    with torch.no_grad():
        file_list = glob.glob("data/*")
        for file in file_list:
            raw_data = np.load(open(file, "rb"))
            observations = raw_data['obs']
            actions = raw_data['action']
            observations = np.moveaxis(observations, 3, 1)
            observations = torch.tensor(observations, dtype=torch.float, device=device) / 255
            z, recon_data, mu, log_var = model(observations)
            file_name = os.path.splitext(os.path.basename(file))[0]
            np.savez_compressed('series/' + file_name, obs=z.cpu().numpy(), action=actions)
