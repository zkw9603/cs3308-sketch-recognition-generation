# src/generation/sketch_rnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=512, latent_size=128, num_layers=2):
        super(SketchRNN, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.label_embedding = nn.Embedding(num_classes, 64)
        
        self.encoder_rnn = nn.GRU(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_log_var = nn.Linear(hidden_size, latent_size)

        self.decoder_rnn = nn.GRU(input_size=3 + latent_size + 64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 3)
    
    def encode(self, x):
        _, h = self.encoder_rnn(x)
        h = h[-1]
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        log_var = torch.clamp(log_var, min=-5.0, max=2.0)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels, max_seq_len, device):
        batch_size = z.size(0)
        label_embed = self.label_embedding(labels)
        
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        outputs = []
        
        decoder_input = torch.zeros(batch_size, 3 + self.latent_size + 64).to(device)
        
        for t in range(max_seq_len):
            output, h = self.decoder_rnn(decoder_input.unsqueeze(1), h)
            output = self.fc_out(output.squeeze(1))
            outputs.append(output)
            
            if t < max_seq_len - 1:
                decoder_input = torch.cat([output, z, label_embed], dim=1)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def forward(self, x, labels, max_seq_len):
        device = x.device
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_batch = self.decode(z, labels, max_seq_len, device)
        
        recon_loss = F.mse_loss(recon_batch, x, reduction='none').mean(dim=(1, 2))
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        return recon_batch, mu, log_var, recon_loss, kld_loss