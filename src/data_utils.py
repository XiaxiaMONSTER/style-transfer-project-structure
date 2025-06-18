import os
import random
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from diffusers import AutoencoderKL

def import_style_images(source_type="example", source_path=None, target_dir="data/style_images"):
    """
    从不同来源导入风格图片
    """
    # 清空目标目录
    os.makedirs(target_dir, exist_ok=True)
    for file in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    if source_type == "example":
        print("使用示例风格图片...")
        # 实际项目中应替换为本地示例图片或提供下载链接
        print("请手动下载示例图片并放入style_images目录")
        
    elif source_type == "kaggle":
        if not source_path:
            raise ValueError("从Kaggle数据集导入时，必须提供source_path")
        print(f"从Kaggle数据集 {source_path} 导入风格图片...")
        shutil.copytree(source_path, target_dir, dirs_exist_ok=True)
        
    elif source_type == "upload":
        if not source_path:
            raise ValueError("从工作区导入时，必须提供source_path")
        print(f"从工作区 {source_path} 导入风格图片...")
        for file in os.listdir(source_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy2(os.path.join(source_path, file), target_dir)
        
    else:
        raise ValueError(f"不支持的source_type: {source_type}")
    
    style_count = len([f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"成功导入 {style_count} 张风格图片到 {target_dir}")
    return style_count

def import_input_image(source_path=None, target_dir="data/input_image"):
    """导入输入图片"""
    os.makedirs(target_dir, exist_ok=True)
    for file in os.listdir(target_dir):
        file_path = os.path.join(target_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    if source_path:
        print(f"从 {source_path} 导入输入图片...")
        shutil.copy2(source_path, os.path.join(target_dir, "input.jpg"))
    else:
        print("使用示例输入图片...")
        # 实际项目中应替换为本地示例图片或提供下载链接
        print("请手动下载示例图片并放入input_image目录")
    
    input_count = len([f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"成功导入 {input_count} 张输入图片到 {target_dir}")
    return input_count

def display_images(image_dir, title, num_images=5):
    """显示图片"""
    plt.figure(figsize=(15, 3))
    plt.suptitle(title, fontsize=16)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:num_images]
    
    if not image_files:
        print(f"在 {image_dir} 中没有找到图片")
        return
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        plt.subplot(1, min(num_images, len(image_files)), i+1)
        plt.imshow(img)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

class StyleDataset(Dataset):
    def __init__(self, image_dir, prompt, vae, size=512, device="cuda"):
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                           if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.prompt = prompt
        self.vae = vae
        self.size = size
        self.device = device
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载并预处理图像
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # 转换为tensor并移动到设备
        image_tensor = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).to(self.device)
        
        # 使用vae编码
        with torch.no_grad():
            # 临时转换为FP16进行VAE编码以节省显存
            image_tensor_fp16 = image_tensor.to(torch.float16)
            latents = self.vae.encode(image_tensor_fp16.unsqueeze(0)).latent_dist.sample()
            # 根据设备能力决定返回格式
            latents = latents.to(torch.float32)  # 保持FP32避免梯度问题
        
        # 按标准缩放
        latents = latents * 0.18215
        
        # 数据增强
        if random.random() < 0.5:
            latents = torch.flip(latents, dims=[-1])
            
        return {"latents": latents.squeeze(0), "prompt": self.prompt}
    