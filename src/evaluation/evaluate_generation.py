# src/evaluation/evaluate_generation.py

import os
import torch
import numpy as np
from src.recognition.model import SketchANet
from src.generation.sketch_rnn import SketchRNN
from src.generation.dataset import QuickDrawDataset
from torchvision import transforms
from PIL import Image

class Binarize(object):
    def __call__(self, img):
        return (img > 0.1).float()

def strokes_to_image(strokes, image_size=64):
    """将笔画序列转换为图像"""
    canvas_size = image_size * 4
    img = np.ones((canvas_size, canvas_size), dtype=np.float32)
    
    x, y = canvas_size // 2, canvas_size // 2
    points = [(x, y)]
    
    for dx, dy, pen in strokes:
        if pen < 0.5:  # 抬笔阈值
            break
        x = int(x + dx * canvas_size / 2)
        y = int(y + dy * canvas_size / 2)
        points.append((x, y))
    
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        x1, y1 = max(0, min(canvas_size-1, x1)), max(0, min(canvas_size-1, y1))
        x2, y2 = max(0, min(canvas_size-1, x2)), max(0, min(canvas_size-1, y2))
        img[y1, x1] = 0.0
        img[y2, x2] = 0.0
    
    pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    return pil_img.resize((image_size, image_size), Image.LANCZOS)

def evaluate_generated_sketches():
    device = torch.device('cpu')
    class_names = ['apple', 'bus', 'cake', 'cat', 'fish']
    num_classes = len(class_names)
    
    # 加载识别模型
    recognition_model = SketchANet(num_classes=num_classes).to(device)
    recognition_model.load_state_dict(torch.load(
        r"E:\机器学习\proj\checkpoints\sketch_a_net_5cls.pth", 
        map_location=device
    ))
    recognition_model.eval()
    
    # 加载生成模型
    generation_model = SketchRNN(num_classes=num_classes).to(device)
    generation_model.load_state_dict(torch.load(
        r"E:\机器学习\proj\checkpoints\sketch_rnn_final.pth",
        map_location=device
    ))
    generation_model.eval()
    
    # 加载真实数据
    dataset = QuickDrawDataset(
        r"E:\机器学习\proj\QuickDraw_generation\selected_classes",
        max_seq_len=100
    )
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        Binarize(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize(mean=[0.0], std=[1.0])
    ])
    
    class_samples = {i: [] for i in range(num_classes)}
    for i in range(len(dataset)):
        if len(class_samples[dataset.labels[i]]) < 20:
            class_samples[dataset.labels[i]].append(i)
    
    total_correct = 0
    total_samples = 0
    results = {cls: {'correct': 0, 'total': 0} for cls in class_names}
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            class_name = class_names[class_idx]
            print(f"Evaluating {class_name}...")
            for sample_idx in class_samples[class_idx]:
                real_stroke, real_label = dataset[sample_idx]
                real_tensor = torch.tensor(real_stroke).unsqueeze(0).to(device)
                label_tensor = torch.tensor([class_idx], dtype=torch.long).to(device)
                
                recon_stroke, _, _, _, _ = generation_model(real_tensor, label_tensor, 100)
                recon_stroke = recon_stroke[0].cpu().numpy()
                
                try:
                    pil_img = strokes_to_image(recon_stroke)
                    input_tensor = transform(pil_img).unsqueeze(0).to(device)
                    
                    output = recognition_model(input_tensor)
                    pred = output.argmax(dim=1).item()
                    
                    total_samples += 1
                    results[class_name]['total'] += 1
                    if pred == class_idx:
                        total_correct += 1
                        results[class_name]['correct'] += 1
                except Exception as e:
                    print(f"Error processing {class_name}: {e}")
                    continue
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n Overall Reconstruction Accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")
    
    for cls, stats in results.items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            print(f"{cls}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    return overall_acc, results

if __name__ == "__main__":
    evaluate_generated_sketches()