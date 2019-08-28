import torch
from torch import nn, optim
from torch.nn import functional as F
import math
import numpy as np


class MDMRNN(nn.Module):
    def __init__(self, num_mixtures, hidden_size,  input_size, num_layers, batch_size, sequence_len, output_size):
        super(MDMRNN, self).__init__()
        self.input_size = input_size
        self.num_mixtures = num_mixtures
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.sequence_len = sequence_len
        self.output_size = output_size
        self.dense = nn.Linear(self.hidden_size, self.output_size * num_mixtures * 3)
    
    def forward(self, x, hidden_state, cell_state):
        x = x.view(self.sequence_len, self.batch_size, self.input_size)
        z, hidden_states = self.lstm(x, (hidden_state, cell_state))
        hidden = hidden_states[0]
        cell = hidden_states[1]
        z = z.view(-1, self.hidden_size)
        z = self.dense(z)
        z = z.view(-1, self.num_mixtures * 3)
        out_logmix, out_mean, out_log_std = get_mdn_coef(z, self.num_mixtures)
        return out_logmix, out_mean, out_log_std, hidden, cell


def rnn_loss(y, out_logmix, out_mean, out_log_std):
    y = y.view(-1, 1)
    loss_func = get_loss_func(out_logmix, out_mean, out_log_std, y)
    return loss_func


def clip_grads(parameters):
    for p in parameters:
        p.grad.data.clamp(-1.0, 1.0)


def reparameterize(log_mix, mean, log_std):
    epsilon = torch.randn_like(log_std.exp())
    recon = torch.sum(log_mix.exp() * (mean + log_std.exp() * epsilon), dim=1)
    recon = recon.view(-1, 32)
    return recon


def log_normal(y, mean, log_std):
    return -0.5 * ((y - mean) / log_std.exp()) ** 2 - log_std - log_std.new([np.log(np.sqrt(2.0 * np.pi))])


def get_loss_func(log_mix, mean, log_std, y):
    v = log_mix + log_normal(y, mean, log_std)
    v = torch.log(torch.sum(v.exp(), dim=1, keepdim=True))
    return - v.mean()


def get_mdn_coef(output, num_mixtures):
    log_mix, mean, log_std = torch.split(output, num_mixtures, dim=1)
    log_mix = log_mix - torch.log(torch.sum(log_mix.exp(), dim=1, keepdim=True))
    return log_mix, mean, log_std
