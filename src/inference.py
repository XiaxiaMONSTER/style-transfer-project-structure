import os
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from .model_utils import apply_style_to_image

def main():
    # 加载配置
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载训练配置
    try:
        with open(os.path.join(config['model']['output_dir'], 'config.yaml'), 'r') as f:
            train_config = yaml.safe_load(f)
    except FileNotFoundError:
        train_config = config
    
    print("=" * 50)
    print("开始风格迁移")
    print("=" * 50)
    
    # 应用风格迁移到输入图片
    input_image_path = os.path.join(config['data']['input_image_dir'], 'input.jpg')
    output_image_path = os.path.join(config['data']['output_images_dir'], 'styled_output.jpg')
    style_model_path = os.path.join(config['model']['output_dir'], 'final_model')
    
    result_image = apply_style_to_image(
        input_image_path=input_image_path,
        style_model_path=style_model_path,
        output_path=output_image_path,
        base_model=config['model']['base_model'],
        prompt=config['inference']['prompt'],
        strength=config['inference']['strength'],
        guidance_scale=config['inference']['guidance_scale'],
        num_inference_steps=config['inference']['num_inference_steps']
    )
    
    # 显示原始输入和风格迁移后的结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("原始图像", fontsize=14)
    plt.imshow(Image.open(input_image_path))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("风格迁移图像", fontsize=14)
    plt.imshow(result_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['data']['output_images_dir'], 'comparison.jpg'))
    plt.show()
    
    print("=" * 50)
    print("测试不同风格强度")
    print("=" * 50)
    
    # 调整参数进行不同效果的风格迁移
    results = []
    
    for strength in config['inference']['test_strengths']:
        print(f"生成风格强度为 {strength} 的图像...")
        output_path = os.path.join(config['data']['output_images_dir'], f'styled_strength_{strength}.jpg')
        result = apply_style_to_image(
            input_image_path=input_image_path,
            style_model_path=style_model_path,
            output_path=output_path,
            base_model=config['model']['base_model'],
            prompt=config['inference']['prompt'],
            strength=strength,
            guidance_scale=config['inference']['guidance_scale'],
            num_inference_steps=config['inference']['num_inference_steps']
        )
        results.append(result)
    
    # 显示不同强度的结果
    plt.figure(figsize=(15, 5))
    plt.suptitle("不同风格强度效果对比", fontsize=16)
    
    plt.subplot(1, len(config['inference']['test_strengths'])+1, 1)
    plt.title("原始图像")
    plt.imshow(Image.open(input_image_path))
    plt.axis('off')
    
    for i, (strength, result) in enumerate(zip(config['inference']['test_strengths'], results)):
        plt.subplot(1, len(config['inference']['test_strengths'])+1, i+2)
        plt.title(f"强度: {strength}")
        plt.imshow(result)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['data']['output_images_dir'], 'strength_comparison.jpg'))
    plt.show()
    
    print("所有任务完成！")
    print("生成的文件：")
    print(os.listdir(config['data']['output_images_dir']))

if __name__ == "__main__":
    main()
    