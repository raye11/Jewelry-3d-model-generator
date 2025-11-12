import torch
import os
from diffusers import StableDiffusionPipeline
from safetensors.torch import save_file

def convert_diffusers_to_safetensors():
    # 输入和输出路径
    diffusers_path = "models/Stable-diffusion/chinese_jewelry_fintune"
    output_path = "models/Stable-diffusion/chinese_jewelry_fintune.safetensors"
    
    print("正在加载Diffusers模型...")
    
    try:
        # 加载diffusers模型
        pipe = StableDiffusionPipeline.from_pretrained(
            diffusers_path,
            torch_dtype=torch.float32,
            safety_checker=None,
            use_safetensors=False,  # 允许加载.bin文件
            local_files_only=True
        )
        
        print("模型加载成功！开始转换...")
        
        # 构建状态字典
        state_dict = {}
        
        # 添加UNet权重
        for key, value in pipe.unet.state_dict().items():
            state_dict[f"model.diffusion_model.{key}"] = value
        
        # 添加VAE权重  
        for key, value in pipe.vae.state_dict().items():
            state_dict[f"first_stage_model.{key}"] = value
        
        # 添加Text Encoder权重
        for key, value in pipe.text_encoder.state_dict().items():
            state_dict[f"cond_stage_model.{key}"] = value
        
        print(f"合并了 {len(state_dict)} 个参数")
        
        # 保存为safetensors
        save_file(state_dict, output_path)
        
        file_size = os.path.getsize(output_path) / 1024**3
        print(f"转换完成！文件大小: {file_size:.2f} GB")
        print(f"保存位置: {output_path}")
        
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    convert_diffusers_to_safetensors()