# 基于LoRA的艺术风格迁移项目

## 项目简介
本项目使用LoRA技术微调Stable Diffusion模型，实现将中国传统绘画风格应用到现代图像上的功能。项目包含数据准备、模型训练、风格迁移和结果可视化等完整流程。

## 功能特点
- 使用LoRA技术高效微调Stable Diffusion模型
- 支持自定义风格图像训练
- 提供多种风格强度调整选项
- 可视化原始图像与风格迁移结果对比

## 环境要求
- Python 3.8+
- PyTorch 2.6.0
- CUDA 12.1+
- 至少12GB GPU内存

## 安装依赖pip install -r requirements.txt
## 使用方法
1. 准备风格图像和输入图像
2. 配置`config.yaml`文件
3. 执行训练脚本：`python train.py`
4. 执行推理脚本：`python inference.py`

## 项目结构.
├── data/                   # 数据目录
│   ├── input_image/        # 输入图像
│   └── style_images/       # 风格图像
├── models/                 # 模型保存目录
├── outputs/                # 输出结果目录
├── src/                    # 源代码目录
│   ├── data_utils.py       # 数据处理工具
│   ├── model_utils.py      # 模型处理工具
│   ├── train.py            # 训练脚本
│   └── inference.py        # 推理脚本
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖文件
└── README.md               # 项目说明    