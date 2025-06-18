import os
import yaml
from .data_utils import import_style_images, import_input_image, display_images
from .model_utils import train_style_model

def main():
    # 加载配置
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 数据准备
    print("=" * 50)
    print("开始数据准备")
    print("=" * 50)
    
    style_count = import_style_images(
        source_type=config['data']['style_source_type'],
        source_path=config['data']['style_source_path'],
        target_dir=config['data']['style_images_dir']
    )
    
    input_count = import_input_image(
        source_path=config['data']['input_image_path'],
        target_dir=config['data']['input_image_dir']
    )
    
    # 显示风格图片
    display_images(config['data']['style_images_dir'], '风格图片')
    
    # 显示输入图片
    display_images(config['data']['input_image_dir'], '输入图片', num_images=1)
    
    # 模型训练
    print("=" * 50)
    print("开始训练风格模型")
    print("=" * 50)
    
    style_model_path = train_style_model(
        style_images_dir=config['data']['style_images_dir'],
        output_dir=config['model']['output_dir'],
        base_model=config['model']['base_model'],
        style_name=config['model']['style_name'],
        num_epochs=config['train']['num_epochs'],
        learning_rate=config['train']['learning_rate'],
        batch_size=config['train']['batch_size']
    )
    
    print(f"模型训练完成，保存路径: {style_model_path}")
    
    # 保存配置
    with open(os.path.join(config['model']['output_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    main()
    