# src/generation/generate.py

import os
import torch
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.sketch_rnn import SketchRNN
from utils.visualize import draw_sketch, draw_sketches_grid

def load_model(checkpoint_path, num_classes, device):
    model = SketchRNN(num_classes=num_classes, hidden_size=512, latent_size=128, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def main():
    device = torch.device('cpu')
    checkpoint_path = r"E:\机器学习\proj\checkpoints\sketch_rnn_final.pth"
    class_names = ['apple', 'bus', 'cake', 'cat', 'fish']
    num_classes = len(class_names)
    
    model = load_model(checkpoint_path, num_classes, device)
    
    # 生成指定类别的草图
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(labels, seq_len=100, device=device)
    
    generated_np = generated.cpu().numpy()
    
    os.makedirs(r"E:\机器学习\proj\figures", exist_ok=True)
    draw_sketches_grid(generated_np, class_names, labels.tolist(), 
                      save_path=r"E:\机器学习\proj\figures\generated_samples.png")
    print("生成样本已保存到 'figures/generated_samples.png'")
    
    # 潜空间插值（apple -> cat）
    z1 = torch.randn(1, 128)
    z2 = torch.randn(1, 128)
    interpolations = []
    interp_labels = []
    
    for i in range(5):
        alpha = i / 4.0
        z_interp = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            sketch = model.generate(torch.tensor([0]), seq_len=100, z=z_interp, device=device)
        interpolations.append(sketch[0].cpu().numpy())
        interp_labels.append(0)
    
    draw_sketches_grid(interpolations, class_names, interp_labels,
                      save_path=r"E:\机器学习\proj\figures\interpolation.png")
    print("潜空间插值结果已保存到 'figures/interpolation.png'")

if __name__ == "__main__":
    main()