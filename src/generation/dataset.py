# src/generation/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, max_seq_len=100, scale_factor=255.0, max_samples_per_class=3840):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.scale_factor = scale_factor
        
        self.data = []
        self.labels = []
        self.class_names = []
        
        class_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        class_files.sort()
        
        for label, class_file in enumerate(class_files):
            class_name = os.path.splitext(class_file)[0]
            self.class_names.append(class_name)
            
            file_path = os.path.join(data_dir, class_file)
            with np.load(file_path, allow_pickle=True, encoding='latin1') as loaded_data:
                train_data = loaded_data['train']
                valid_data = loaded_data['valid']
                test_data = loaded_data['test']
                all_data = np.concatenate([train_data, valid_data, test_data], axis=0)
                
                selected_data = all_data[:max_samples_per_class]
                
                for seq in selected_data:
                    if len(seq) > 1:
                        self.data.append(seq)
                        self.labels.append(label)
        
        print(f" Loaded {len(self.data)} sequences from {len(self.class_names)} classes: {self.class_names}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        raw_seq = self.data[idx].astype(np.float32)
        label = self.labels[idx]
        
        x = raw_seq[:, 0]
        y = raw_seq[:, 1]
        
        x = x / self.scale_factor
        y = y / self.scale_factor
        
        x_padded = np.concatenate([[0.0], x]).astype(np.float32)
        y_padded = np.concatenate([[0.0], y]).astype(np.float32)
        dx = np.diff(x_padded)
        dy = np.diff(y_padded)
        
        pen = np.ones(len(dx), dtype=np.float32)
        pen[-1] = 0.0
        
        stroke = np.stack([dx, dy, pen], axis=1)
        
        if len(stroke) < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - len(stroke), 3), dtype=np.float32)
            stroke = np.vstack([stroke, padding])
        else:
            stroke = stroke[:self.max_seq_len]
        
        return torch.from_numpy(stroke), torch.tensor(label, dtype=torch.long)