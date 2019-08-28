from models import rnn
import numpy as np
import os
import torch
from torch import nn, optim
import glob
from hyperparameters import *


def init_hidden(num_layers, batch_size, hidden_size, device):
    hidden = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    cell = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    return hidden, cell


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn_model = rnn.MDMRNN(MDN_NUM_MIXTURES, MDN_HIDDEN_SIZE, MDN_INPUT_SIZE, MDN_NUM_LAYERS, MDN_BATCH_SIZE, MDN_SEQUENCE_LEN, MDN_OUTPUT_SIZE)

    if os.path.exists("checkpoints/rnn_checkpoint.pth"):
        rnn_model.load_state_dict(torch.load("checkpoints/rnn_checkpoint.pth"))
    rnn_model.to(device)
    rnn_model = rnn_model.train()
    
    optimizer = optim.Adam(rnn_model.parameters(), lr=MDN_LEARNING_RATE)

    data_files = glob.glob('series/*')

    i = 0
    for epoch in range(MDN_EPOCHS):
        for file in data_files:
            raw_data = np.load(open(file, "rb"))
            states = raw_data['obs'][:1000]
            actions = raw_data['action'][:1000]
            target_states = raw_data['obs'][1:]  # TODO Change to -1 later
            states = torch.tensor(states, device=device)
            actions = torch.tensor(actions, device=device)
            target_states = torch.tensor(target_states, device=device)
            target_states = target_states.view(-1, 1)
            optimizer.zero_grad()
            hidden_state, cell_state = init_hidden(MDN_NUM_LAYERS, MDN_BATCH_SIZE, MDN_HIDDEN_SIZE, device)
            actions = actions.type(torch.float32)
            inputs = torch.cat((states, actions), dim=1)
            pi, mean, sigma, hidden_state, cell_state = rnn_model(inputs, hidden_state, cell_state)
            target_states = target_states.view(MDN_SEQUENCE_LEN, MDN_BATCH_SIZE, -1, 1)
            loss = rnn.rnn_loss(target_states, pi, mean, sigma)
            loss.backward()
            rnn.clip_grads(rnn_model.parameters())
            optimizer.step()
            loss = loss.item()
            i += 1
            if i % 100 == 0:
                print("Completed Step ", i, "Loss", loss)
                torch.save(rnn_model.state_dict(), "checkpoints/rnn_checkpoint.pth")
