# src/generation/test_dataset.py
from src.generation.dataset import QuickDrawDataset

if __name__ == "__main__":
    dataset = QuickDrawDataset(r"E:\机器学习\proj\QuickDraw_generation\selected_classes")
    print("Dataset size:", len(dataset))
    seq, label = dataset[0]
    print("Sequence shape:", seq.shape, "Label:", label)
    print("First 5 strokes:", seq[:5])