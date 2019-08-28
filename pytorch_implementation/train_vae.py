import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/train_vae.log', level=logging.INFO)
from models import vae
import numpy as np
import os
import torch
from torch import nn, optim
import glob
from hyperparameters import *
import time


# Similar to function from original experiment. Load all files into memory to speed up training.
def create_dataset(file_list_):
    print("inside create_dataset")
    start = time.time()
    n = len(file_list_)
    m = SEQUENCE_LENGTH
    data = np.zeros((m * n, 64, 64, 3), dtype=np.uint8)
    idx = 0
    for file in file_list_:
        raw_data = np.load(open(file, "rb"))['obs']
        raw_data = raw_data[:m]
        data[idx:idx + m] = raw_data
        idx += m
    np.random.shuffle(data)
    data = np.moveaxis(data, 3, 1)  # Reshape so that channels first
    end = time.time()
    logging.info("Data size" + str(data.shape))
    logging.info("Time taken to load dataset: " + str(end - start))
    return data


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = vae.ConvVAE(VAE_Z_SIZE, VAE_KL_TOLERANCE)
    if os.path.exists("checkpoints/vae_checkpoint.pth"):
        model.load_state_dict(torch.load("checkpoints/vae_checkpoint.pth"))
    model.to(device)
    model = model.train()
    optimizer = optim.Adam(model.parameters(), lr=VAE_LEARNING_RATE)

    file_list = glob.glob("data/*")
    # Split file list into chunks to reduce memory consumption
    file_lists = [file_list[i * VAE_NO_OF_FILES:(i + 1) * VAE_NO_OF_FILES] for i in range((len(file_list) + VAE_NO_OF_FILES - 1) // VAE_NO_OF_FILES)]
    i = 0
    for epoch in range(VAE_EPOCHS):
        start = time.time()
        for file_list in file_lists:
            dataset = create_dataset(file_list)
            total_length = len(dataset)
            num_batches = int(np.floor(total_length / VAE_BATCH_SIZE))
            for idx in range(num_batches):
                batch = dataset[idx * VAE_BATCH_SIZE:(idx + 1) * VAE_BATCH_SIZE]
                batch = torch.tensor(batch, dtype=torch.float, device=device) / 255
                optimizer.zero_grad()
                z, recon_data, mu, log_var = model(batch)
                loss = vae.vae_loss(recon_data, batch, mu, log_var, model.kl_tolerance, model.z_size)
                loss.backward()
                optimizer.step()
                loss = loss.item()
                i += 1
                if i % 100 == 0:
                    print("Loss at batch:", str(i), " is: ", str(loss))
            torch.save(model.state_dict(), "checkpoints/vae_checkpoint.pth")

        end = time.time()
        logging.info("Time taken for epoch " + str(epoch) + ": " + str(end - start))
