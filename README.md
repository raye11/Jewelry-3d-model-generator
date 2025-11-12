## 🌟 项目简介

本项目是基于AIGC技术的少数民族首饰数字化传承与创新设计平台。通过融合前沿的Stable Diffusion、Hunyuan3D、TripoSR等AI模型，我们实现了从文本描述到高质量3D首饰模型的端到端生成，致力于解决少数民族非遗工艺的传承危机，推动文化创新。

## 🎯 核心功能

- **智能文生3D**：输入民族文化描述，自动生成对应的3D首饰模型
- **多模态生成**：支持文本→图像→3D模型的完整生成流程
- **文化精准性**：针对苗族银饰、藏族珊瑚、彝族漆器等特定民族风格优化
- **可制造输出**：生成水密、可3D打印的高质量网格模型

## 🛠️ 技术架构
文化描述 → DeepSeek大模型 → 精准提示词 → SDXL生成 → Hunyuan3D/TripoSG → 3D模型
### 核心技术栈
- **文本理解**：DeepSeek
- **图像生成**：Stable Diffusion XL (SDXL)
- **3D重建**：Hunyuan3D-2.1、TripoSG

快速开始
克隆项目：

bash
git clone https://github.com/raye11/Jewelry-3d-model-generator.git
cd Jewelry-3d-model-generator

🎨 使用示例
输入文本描述：

text
"一枚彝族风格的戒指"
输出结果：

生成符合彝族文化特征的3戒指模型

可进行面数简化

可导出为.glb格式用于3D打印


Stable Diffusion XL
bibtex
@software{rombach2021highresolution,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
  year={2021},
  url={https://github.com/Stability-AI/stablediffusion}
}
TripoSR
bibtex
@article{triposr2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tsiriklis, Dimitrios and others},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024},
  url={https://github.com/VAST-AI-Research/TripoSR}
}
Hunyuan3D-2.1
bibtex
@software{tencent2024hunyuan3d,
  title={Hunyuan3D-2.1: Advanced 3D Generation Model},
  author={Tencent},
  year={2024},
  url={https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1}
}

📄 许可证
本项目采用 MIT 许可证 - 详见 LICENSE 文件。

注意：本项目使用的第三方模型受其各自许可证约束：

Stable Diffusion XL: CreativeML Open RAIL++-M License

Hunyuan3D-2.1: Tencent Hunyuan 3D 2.1 Community License

TripoSR: Apache License 2.0

🤝 致谢
感谢以下开源项目为本研究提供的技术支持：

Stability AI - 提供Stable Diffusion XL模型

Tencent Hunyuan - 提供Hunyuan3D-2.1模型

VAST AI Research - 提供TripoSG模型

AUTOMATIC1111 - 提供WebUI框架

📞 联系我们
如有问题或合作意向，请通过以下方式联系：

GitHub Issues: 项目Issues页面

邮箱: 1024128103@qq.com

