import torch
from PIL import Image
import sys
import os
from pathlib import Path

# 直接添加模型路径到系统路径
model_path = r"D:\work\AUTO1111\webui\models\Zero123\zero123plus-v1.1"
sys.path.insert(0, model_path)

# 直接导入推理代码
inference_file = os.path.join(model_path, "inference.py")
if not os.path.exists(inference_file):
    print(f"❌ 推理文件不存在: {inference_file}")
    sys.exit(1)

# 动态导入推理模块
import importlib.util
spec = importlib.util.spec_from_file_location("inference", inference_file)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)

print("✅ 成功导入推理代码")

def test_direct_inference():
    """直接使用原始推理代码测试"""
    print("🎯 直接推理测试...")
    
    try:
        # 加载图像
        test_image_path = r"D:\work\AUTO1111\webui\1.png"
        if not os.path.exists(test_image_path):
            print("❌ 测试图像不存在")
            return False
            
        image = Image.open(test_image_path).convert("RGB")
        image = image.resize((256, 256))
        print(f"输入图像: {image.size}")
        
        # 直接使用原始推理代码
        print("使用原始 Zero123PlusPipeline...")
        
        # 从模型目录加载管道
        pipeline = inference.Zero123PlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            print("✅ 使用 GPU")
        
        # 准备管道（调用 prepare 方法）
        if hasattr(pipeline, 'prepare'):
            pipeline.prepare()
            print("✅ 管道准备完成")
        
        # 生成测试
        print("生成右侧视角...")
        result = pipeline(
            image=image,
            elevation=0,
            azimuth=90,
            num_inference_steps=15,  # 较少的步数用于测试
            guidance_scale=3.0,
            height=256,
            width=256,
            output_type="pil"
        )
        
        # 保存结果
        output_path = "direct_test_output.jpg"
        if hasattr(result, 'images') and result.images:
            result.images[0].save(output_path)
            print(f"✅ 生成完成: {output_path}")
            return True
        else:
            print("❌ 没有生成图像")
            return False
            
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_components():
    """简单组件测试"""
    print("\n🔧 简单组件测试...")
    
    try:
        # 测试直接加载组件
        from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
        from diffusers import AutoencoderKL, UNet2DConditionModel, EulerAncestralDiscreteScheduler
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print("1. 测试 VAE 加载...")
        vae = AutoencoderKL.from_pretrained(
            os.path.join(model_path, "vae"),
            local_files_only=True
        ).to(device, dtype=dtype)
        print("   ✅ VAE 加载成功")
        
        print("2. 测试 UNet 加载...")
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(model_path, "unet"),
            local_files_only=True
        ).to(device, dtype=dtype)
        print("   ✅ UNet 加载成功")
        
        print("3. 测试文本编码器加载...")
        text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(model_path, "text_encoder"),
            local_files_only=True
        ).to(device, dtype=dtype)
        print("   ✅ 文本编码器加载成功")
        
        print("4. 测试视觉编码器加载...")
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
            os.path.join(model_path, "vision_encoder"),
            local_files_only=True
        ).to(device, dtype=dtype)
        print("   ✅ 视觉编码器加载成功")
        
        print("🎉 所有组件加载测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 组件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎪 Zero-1-to-3 直接测试工具")
    print(f"模型路径: {model_path}")
    
    # 检查模型目录
    if not os.path.exists(model_path):
        print("❌ 模型路径不存在")
        return
    
    # 1. 先测试组件加载
    print("\n=== 阶段1: 组件加载测试 ===")
    if not test_simple_components():
        print("💥 组件加载测试失败")
        return
    
    # 2. 测试完整推理
    print("\n=== 阶段2: 完整推理测试 ===")
    if test_direct_inference():
        print("🎉 完整测试成功！")
    else:
        print("💥 推理测试失败")

if __name__ == "__main__":
    main()