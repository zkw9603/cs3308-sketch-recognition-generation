# utils/visualize.py

import os
import numpy as np
from PIL import Image, ImageDraw

def draw_sketch(strokes, image_size=64, save_path=None, title="Sketch"):
    canvas_size = image_size * 4
    img = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    
    x, y = canvas_size // 2, canvas_size // 2
    points = [(x, y)]
    
    for dx, dy, pen in strokes:
        x += dx * 50
        y += dy * 50
        
        x_clipped = max(0, min(canvas_size - 1, int(x)))
        y_clipped = max(0, min(canvas_size - 1, int(y)))
        points.append((x_clipped, y_clipped))
        
        if pen < 0.5:
            break
    
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=0, width=2)
    
    img = img.resize((image_size, image_size), Image.LANCZOS)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
    
    return img

def draw_sketches_grid(sketches, class_names, labels, save_path=None, cols=5):
    import matplotlib.pyplot as plt
    
    rows = (len(sketches) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i in range(len(sketches)):
        pil_img = draw_sketch(sketches[i])
        axes[i].imshow(pil_img, cmap='gray')
        axes[i].set_title(f"{class_names[labels[i]]}")
        axes[i].axis('off')
    
    for i in range(len(sketches), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()