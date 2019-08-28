import torch
from torch import nn, optim
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(self, z_size, kl_tolerance):
        super(ConvVAE, self).__init__()
        self.z_size = z_size
        self.kl_tolerance = kl_tolerance
        # Encoder
        self.conv_1 = nn.Conv2d(3, 32, 4, 2)
        self.conv_2 = nn.Conv2d(32, 64, 4, 2)
        self.conv_3 = nn.Conv2d(64, 128, 4, 2)
        self.conv_4 = nn.Conv2d(128, 256, 4, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, self.z_size)
        self.fc2 = nn.Linear(256 * 2 * 2, self.z_size)
        
        # Decoder
        self.fc3 = nn.Linear(self.z_size, 1024)
        self.conv_t_1 = nn.ConvTranspose2d(1024, 128, 5, 2)
        self.conv_t_2 = nn.ConvTranspose2d(128, 64, 5, 2)
        self.conv_t_3 = nn.ConvTranspose2d(64, 32, 6, 2)
        self.conv_t_4 = nn.ConvTranspose2d(32, 3, 6, 2)
    
    def encode(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = x.view(-1, 256 * 2 * 2)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var / 2.0)
        epsilon = torch.randn_like(mu)
        return mu + sigma * epsilon
    
    def decode(self, z):
        z = self.fc3(z)
        z = z.view(-1, 1024, 1, 1)
        z = F.relu(self.conv_t_1(z))
        z = F.relu(self.conv_t_2(z))
        z = F.relu(self.conv_t_3(z))
        z = torch.sigmoid(self.conv_t_4(z))
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var


def vae_loss(recon_x, x, mu, log_var, kl_tolerance, z_size):
    r_loss = torch.sum((recon_x - x).pow(2), dim=(1, 2, 3))
    r_loss = torch.mean(r_loss)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = torch.max(kl_loss, kl_loss.new([kl_tolerance * z_size]))
    kl_loss = torch.mean(kl_loss)
    return r_loss + kl_loss
