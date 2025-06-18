import os
import matplotlib.pyplot as plt
from PIL import Image

def visualize_results(output_dir="outputs"):
    """可视化风格迁移结果"""
    # 确保目录存在
    if not os.path.exists(output_dir):
        print(f"输出目录 {output_dir} 不存在")
        return
    
    # 查找结果图像
    result_files = sorted([f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not result_files:
        print(f"在 {output_dir} 中没有找到结果图像")
        return
    
    # 创建网格布局
    n = len(result_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(5*cols, 5*rows))
    
    for i, file in enumerate(result_files):
        img_path = os.path.join(output_dir, file)
        img = Image.open(img_path)
        
        plt.subplot(rows, cols, i+1)
        plt.title(file)
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_results()
    