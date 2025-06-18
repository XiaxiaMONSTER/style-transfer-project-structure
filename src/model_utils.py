import os
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from tqdm.auto import tqdm
from .data_utils import StyleDataset

def train_style_model(style_images_dir, output_dir, base_model="runwayml/stable-diffusion-v1-5", 
                      style_name="custom_style", num_epochs=5, learning_rate=1e-5, batch_size=2):
    
    os.makedirs(output_dir, exist_ok=True)
    
    accelerator = Accelerator(
        mixed_precision="no",  # 禁用混合精度
        gradient_accumulation_steps=1
    )
    
    print(f"使用设备: {accelerator.device}")
    print(f"混合精度设置: {accelerator.mixed_precision}")
    
    # 加载基础模型（为了节省显存仍使用FP16，但在训练时转换为FP32）
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, safety_checker=None
    ).to(accelerator.device)
    
    # 启用优化
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("成功启用xformers优化")
    except:
        print("xformers优化失败，使用标准模式")
    
    # LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
        task_type="TEXT_TO_IMAGE"
    )
    
    # 应用LoRA到UNet
    unet = pipe.unet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # 为训练转换UNet为FP32
    unet = unet.to(torch.float32)
    
    # 准备数据集
    pipe.vae = pipe.vae.to(torch.float16)
    dataset = StyleDataset(
        image_dir=style_images_dir,
        prompt=f"{style_name} style, high quality, detailed",
        vae=pipe.vae,
        device=accelerator.device
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )
    
    # 准备训练
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # 训练循环
    print("开始训练...")
    for epoch in range(num_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"训练轮次 {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 前向传播
            latents = batch["latents"].to(torch.float32)  # 确保是FP32
            
            text_inputs = pipe.tokenizer(
                batch["prompt"],
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(accelerator.device)
            
            # 获取文本嵌入并转换为FP32
            with torch.no_grad():
                text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0].to(torch.float32)
            
            # 噪声处理
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, 
                (bsz,), device=latents.device, dtype=torch.long
            )
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # 预测噪声
            with accelerator.accumulate(unet):
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # 计算损失
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # 反向传播
                accelerator.backward(loss)
                
                # 梯度裁剪
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                
                # 优化步骤
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.detach().item()
            progress_bar.set_postfix({"损失": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"训练轮次 {epoch+1}/{num_epochs} - 平均损失: {avg_loss:.4f}")
        
        # 保存检查点
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
            accelerator.wait_for_everyone()
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(save_path)
            print(f"检查点已保存: {save_path}")
    
    # 保存最终模型
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    final_model_path = os.path.join(output_dir, "final_model")
    unwrapped_unet.save_pretrained(final_model_path)
    print(f"训练完成！最终模型保存于: {final_model_path}")
    return final_model_path

def apply_style_to_image(
    input_image_path,
    style_model_path,
    output_path,
    base_model="runwayml/stable-diffusion-v1-5",
    prompt="Chinese traditional painting style, high quality, detailed",
    strength=0.7,
    guidance_scale=8.0,
    num_inference_steps=50
):
    print("加载基础模型...")
    # 加载基础模型
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    
    # 启用优化
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("成功启用xformers优化")
    except:
        print("xformers优化失败，使用标准模式")
    
    print("加载训练好的LoRA模型...")
    # 创建LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
        task_type="TEXT_TO_IMAGE"
    )
    
    # 应用LoRA并加载训练好的权重
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    
    # 加载训练好的LoRA权重
    pipe.unet = PeftModel.from_pretrained(pipe.unet, style_model_path)
    
    print("创建图像到图像的管道...")
    # 转换为Img2Img管道
    img2img_pipe = StableDiffusionImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
        safety_checker=pipe.safety_checker,
        feature_extractor=getattr(pipe, 'feature_extractor', None)
    ).to("cuda")
    
    # 处理输入图像
    print(f"处理输入图像: {input_image_path}")
    input_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    
    # 风格迁移
    print("执行风格迁移...")
    result = img2img_pipe(
        prompt=prompt,
        image=input_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    result.save(output_path)
    print(f"结果已保存: {output_path}")
    return result
    