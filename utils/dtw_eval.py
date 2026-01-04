# utils/dtw_eval.py

import numpy as np
from dtw import dtw
from src.generation.dataset import QuickDrawDataset

def calculate_dtw_distance():
    real_dataset = QuickDrawDataset(
        r"E:\机器学习\proj\QuickDraw_generation\selected_classes",
        max_seq_len=100
    )
    
    distances = []
    num_samples = 50
    
    for i in range(num_samples):
        seq1 = real_dataset[i][0]
        seq2 = real_dataset[(i + 1) % len(real_dataset)][0]
        dist, _, _, _ = dtw(seq1[:, :2], seq2[:, :2], dist=lambda x, y: np.linalg.norm(x - y, ord=2))
        distances.append(dist)
    
    avg_dtw = np.mean(distances)
    print(f"Average DTW Distance between real samples: {avg_dtw:.4f}")
    return avg_dtw

if __name__ == "__main__":
    calculate_dtw_distance()