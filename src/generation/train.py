# src/generation/train.py

import os
import torch
from torch.utils.data import DataLoader
from src.generation.dataset import QuickDrawDataset
from src.generation.sketch_rnn import SketchRNN

def train():
    data_dir = r"E:\机器学习\proj\QuickDraw_generation"
    checkpoint_dir = r"E:\机器学习\proj\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = 64
    num_epochs = 30
    learning_rate = 0.0001
    max_seq_len = 100
    hidden_size = 512
    latent_size = 128
    num_layers = 2

    dataset = QuickDrawDataset(data_dir, max_seq_len=max_seq_len, scale_factor=255.0)


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device('cpu')
    model = SketchRNN(
        num_classes=len(dataset.class_names),
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.train()
    for epoch in range(num_epochs):
        if epoch < 5:
            kl_weight = 0.0
        elif epoch < 15:
            kl_weight = (epoch - 5) * 0.1 
        else:
            kl_weight = 1.0
        
        total_loss = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            recon_batch, mu, log_var, recon_loss, kld_loss = model(data, labels, max_seq_len)
            loss = recon_loss.mean() + kl_weight * kld_loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.mean().item():.4f}, "
                      f"KLD: {kld_loss.mean().item():.4f}, KL_Weight: {kl_weight:.2f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"sketch_rnn_epoch_{epoch+1}.pth"))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "sketch_rnn_final.pth"))
    print("Training finished! Model saved to 'checkpoints/sketch_rnn_final.pth'")

if __name__ == "__main__":
    train()