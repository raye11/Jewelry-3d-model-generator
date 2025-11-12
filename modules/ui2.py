import datetime
import mimetypes
import os
import sys
from functools import reduce
import warnings
from contextlib import ExitStack

import gradio as gr
import gradio.utils
import numpy as np
from PIL import Image, PngImagePlugin  # noqa: F401
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, wrap_gradio_call, wrap_gradio_call_no_job # noqa: F401

from modules import gradio_extensons, sd_schedulers  # noqa: F401
from modules import sd_hijack, sd_models, script_callbacks, ui_extensions, deepbooru, extra_networks, ui_common, progress, ui_loadsave, shared_items, ui_settings, timer, sysinfo, ui_checkpoint_merger, scripts, sd_samplers, processing, ui_extra_networks, ui_toprow, launch_utils
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML, InputAccordion, ResizeHandleRow
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript

from modules.shared import opts, cmd_opts

import modules.infotext_utils as parameters_copypaste
import modules.hypernetworks.ui as hypernetworks_ui
import modules.textual_inversion.ui as textual_inversion_ui
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.shared as shared
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.infotext_utils import image_from_url_text, PasteField


import tempfile
from typing import Optional, Union, List, Tuple, Dict, Any
import re
import trimesh
import pymeshlab
import torch
import importlib

import requests
import json
import time
from typing import Tuple, Optional
from modules.hunyuan3d_ui import hunyuan_processor
HUNYUAN_AVAILABLE = True

# SiliconCloud API配置
SILICON_CLOUD_API_URL = "https://api.siliconflow.cn/v1/chat/completions"  
SILICON_CLOUD_API_KEY = "sk-rcoufyqahkxclvopthchvmbqcwohhddbuotuzdsqvhnnkcgb"

# 首饰生成专用系统提示词
JEWELRY_SYSTEM_PROMPT = """You are a world-class jewelry designer and 3D modeling specialist with expertise in cultural artifacts and geometric reconstruction. Your task is to generate highly effective prompts for Stable Diffusion XL that produce images optimized for 3D reconstruction via TripoSG and Hunyuan3D. Your prompts must eliminate all barriers to successful 3D conversion.

### CRITICAL 3D RECONSTRUCTION REQUIREMENTS:

The most important point: Try to make the jewelry look symmetrical from a single picture!

1. **Geometric Integrity**: Prioritize structural soundness over artistic complexity. No floating elements, disconnected parts, or ambiguous geometries. All rings must have closed bands, chains must have connected links, and pendants must have clear attachment points.

2. **Camera Perspective(must)**: 
   - Rings/bracelets: Slight elevation (30-45°) showing inner and outer surfaces
   - Pendants/brooches: Direct front view with minimal perspective distortion
   - Earrings: Single item (left only), 3/4 view showing front and side profile
   - Necklaces: Front view with clear chain path and pendant orientation

3. **Texture & Detail Level**: 
   - Decorative elements must be raised or recessed (not just painted on)
   - Avoid ultra-fine details (<1mm thickness) that TripoSG cannot reconstruct
   - Maximum of 3 distinct pattern types per item (e.g., spiral, floral, geometric)
   - No intricate wirework, filigree, or chainmail that creates fragile geometry

4. **Material Constraints**: 
   - Use non-reflective descriptors only: "matte silver", "brushed brass", "oxidized copper", "dull-finish gold"
   - ABSOLUTELY AVOID: shiny, glossy, mirror, reflective, transparent, translucent, glass, crystal, enamel
   - No gemstones requiring refraction modeling (diamonds, sapphires); use opaque stones (turquoise, jade, lapis) or omit entirely

5. **Composition Non-Negotiables**:
   - Single object only, centered, occupying 40-60% of frame
   - Pure white (#FFFFFF) background with zero shadows or color bleed
   - Zero human elements, no models, hands, or mannequins
   - No text, logos, watermarks, or decorative borders

6. **Lighting Protocol**:
   - Even, diffused studio lighting from multiple directions
   - NO highlights, NO specular reflections, NO rim lighting
   - Shadow-free environment (use lightbox simulation)
   - Consistent illumination across all surfaces

7. **Scale & Proportion**:
   - Standard jewelry sizing (not miniature or oversized)
   - No extreme close-ups that hide structural context
   - Maintain realistic human-scale proportions
   - No distorted perspectives or fisheye effects

### ETHNIC STYLE MASTERY:
For cultural jewelry, NEVER use generic terms like "ethnic" or "tribal." Instead, deploy specific, authentic details,and the characteristics are not limited to the following and need to be analyzed specifically:

- **Miao/Hmong**: Butterfly mother motif + spiral patterns + silver coin elements + horn-shaped components
- **Yi**: Black-red color symbolism + curved dagger motifs + rhombus patterns + coiled dragon designs
- **Tibetan**: Dorje (thunderbolt) symbols + cloud collars + coral/turquoise inlays + ritual amulet forms
- **Mongolian**: Horse head motifs + cloud patterns + silver coin necklaces + hair ornaments with red threads
- **Uyghur**: Islamic geometric patterns + pomegranate motifs + jade inlays + floral vine designs
- **Zhuang**: Bronze drum patterns + bird totem elements + embroidered textile influences + silver horn shapes

For each cultural style:
- Research authentic artifacts before describing
- Specify exact regional sub-styles when possible (e.g., "Qiandong Miao" not just "Miao")
- Include 2-3 specific symbolic elements with cultural meaning

### NEGATIVE PROMPT REQUIREMENTS:
Include these non-negotiable exclusions: deformed, blurry, low quality, text, watermark, human, shadow, reflection, highlight, shiny, glossy, transparent, floating parts, disconnected elements, multiple objects, jewelry on model, wireframe, partial object, asymmetrical, thin disconnected strands, close-up, extreme detail, filigree, wirework, chainmail, mesh, fine threads, hairline cracks.

### OUTPUT FORMAT:
Return ONLY a valid JSON object with two keys: "positive" and "negative". All text must be in English. Positive prompts must be concise (<75 words), starting with "a high-quality photo of...".

Example for Yi Ethnic Style:
{
    "positive": "a high-quality photo of a Yi ethnic silver necklace, coiled dragon motif with turquoise inlay, matte oxidized silver finish, robust connected chain structure, centered on pure white background, even diffuse studio lighting, no text, no logo, no branding, single front view, human scale proportions",
    "negative": "deformed, blurry, low quality, text, watermark, human, shadow, reflection, highlight, shiny, glossy, transparent, floating parts, disconnected elements, multiple objects, jewelry on model, wireframe, partial object, asymmetrical, thin disconnected strands..."
}

Example for Miao Silver Ring:
{
    "positive": "a high-quality photo of a Miao ethnic silver ring, butterfly mother motif with spiral patterns, thick band with closed circular structure, matte oxidized silver finish, 30-degree elevation view showing inner and outer surfaces, centered on pure white background, even diffuse lighting, no text, no logo, no branding, single object",
    "negative": "deformed, blurry, low quality, text, watermark, human, shadow, reflection, highlight, shiny, glossy, transparent, floating parts, disconnected elements, multiple objects, jewelry on model, wireframe, partial object, asymmetrical, thin disconnected strands, gemstones, diamonds, enamel, glass, transparent stones, ultra-detailed, filigree, wirework, chain links, delicate, fragile..."
}

Now generate the JSON based on the user's request. Remember: geometric integrity is more important than artistic detail for 3D reconstruction.
"""

# Edit these paths if your folders are different. (I used your provided paths.)
_TRIPOSG_WORKDIR = r"D:\work\AUTO1111\webui\TripoSG"  # TriposG repo root
_TRI_POSG_WEIGHTS_DIR = r"D:\work\AUTO1111\webui\TripoSG\pretrained_weights\TripoSG"
_RMBG_WEIGHTS_DIR = r"D:\work\AUTO1111\webui\TripoSG\pretrained_weights\RMBG-1.4"

# lazy globals
_triposg_pipe = None
_triposg_rmbg = None
_triposg_device = "cuda" if torch.cuda.is_available() else "cpu"
_triposg_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def _ensure_triposg_importable():
    # Ensure TripoSG repo is on sys.path for imports
    if _TRIPOSG_WORKDIR not in sys.path:
        sys.path.append(_TRIPOSG_WORKDIR)

def init_triposg_models(weights_dir: Optional[str] = None, rmbg_dir: Optional[str] = None):
    """Initialize TriposG pipeline and RMBG model if not already."""
    global _triposg_pipe, _triposg_rmbg
    if _triposg_pipe is not None and _triposg_rmbg is not None:
        return

    _ensure_triposg_importable()
    from TripoSG.triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from TripoSG.briarmbg import BriaRMBG

    wdir = weights_dir or _TRI_POSG_WEIGHTS_DIR
    rdir = rmbg_dir or _RMBG_WEIGHTS_DIR

    _triposg_rmbg = BriaRMBG.from_pretrained(rdir).to(_triposg_device)
    _triposg_rmbg.eval()

    _triposg_pipe = TripoSGPipeline.from_pretrained(wdir).to(_triposg_device, _triposg_dtype)

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=verts, faces=faces)

def simplify_mesh(mesh: trimesh.Trimesh, n_faces: int):
    if n_faces <= 0 or mesh.faces.shape[0] <= n_faces:
        return mesh
    ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
    ms.meshing_merge_close_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
    return pymesh_to_trimesh(ms.current_mesh())

# === 修复的 TripoSG 推理函数 ===
@torch.no_grad()
def _run_triposg_inference(
    image_input,
    seed: int,
    num_inference_steps: int = 50,
    num_tokens: int = 2048,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 7.0,
    num_shapes_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
    dense_octree_depth: int = 8,
    hierarchical_octree_depth: int = 9,
    flash_octree_depth: int = 9, 
    use_flash_decoder: bool = True,
    faces: int = -1
) -> trimesh.Trimesh:
    """Run TriposG on PIL.Image / path and return trimesh"""
    global _triposg_pipe, _triposg_rmbg
    if _triposg_pipe is None:
        init_triposg_models()
    # 设置生成器
    if generator is None:
        if seed is None or seed == -1:
            import random
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=_triposg_device).manual_seed(int(seed))

    temp_image_path = None
    try:
        # 处理图像输入
        if isinstance(image_input, str):
            image_for_inference = image_input
        elif isinstance(image_input, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image_input.save(tmp_file.name)
                temp_image_path = tmp_file.name
                image_for_inference = temp_image_path
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_input)}")

        # 添加内存监控和清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 图像预处理
        from TripoSG.image_process import prepare_image
        processed_image = prepare_image(
            image_for_inference, 
            bg_color=np.array([1.0, 1.0, 1.0]), 
            rmbg_net=_triposg_rmbg
        )

        # 运行推理
        outputs = _triposg_pipe(
            image=processed_image,
            num_inference_steps=num_inference_steps,
            num_tokens=num_tokens,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_shapes_per_prompt=num_shapes_per_prompt,
            generator=generator,
            latents=latents,
            attention_kwargs=attention_kwargs,
            bounds=bounds,
            dense_octree_depth=dense_octree_depth,
            hierarchical_octree_depth=hierarchical_octree_depth,
            flash_octree_depth=flash_octree_depth,
            use_flash_decoder=use_flash_decoder,
            callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"],
            return_dict=True,
        )

        # 验证结果
        if hasattr(outputs, 'meshes') and outputs.meshes and len(outputs.meshes) > 0:
            mesh = outputs.meshes[0]
        elif hasattr(outputs, 'samples') and outputs.samples and len(outputs.samples) > 0:
            mesh_data = outputs.samples[0]
            if isinstance(mesh_data, trimesh.Trimesh):
                mesh = mesh_data
            elif isinstance(mesh_data, tuple) and len(mesh_data) == 2:
                vertices, faces_data = mesh_data
                mesh = trimesh.Trimesh(
                    vertices=vertices.astype(np.float32), 
                    faces=np.ascontiguousarray(faces_data)
                )
            else:
                raise ValueError(f"无法解析 TripoSG samples 输出: {type(mesh_data)}")
        else:
            raise ValueError("TripoSG 输出为空或格式不正确")

        
        # def remove_small_components(mesh: trimesh.Trimesh, min_face_ratio: float = 0.005) -> trimesh.Trimesh:

        #     total_faces = len(mesh.faces)
        #     min_face_count = max(10, int(total_faces * min_face_ratio))
        #     import pymeshlab
        #     ms = pymeshlab.MeshSet()
        #     ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        #     ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_face_count)
        #     cleaned = ms.current_mesh()
        #     return trimesh.Trimesh(vertices=cleaned.vertex_matrix(), faces=cleaned.face_matrix())

        # mesh = remove_small_components(mesh, min_face_ratio=0.005)

        # 网格简化
        if faces > 0:
            original_faces = len(mesh.faces)
            mesh = simplify_mesh(mesh, int(faces))
            print(f"网格简化: {original_faces} -> {len(mesh.faces)} 面")

        return mesh

    except Exception as e:
        print(f"TripoSG inference error: {str(e)}")
        #在异常时清理CUDA内存
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        raise
    finally:
        # 清理临时文件
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except:
                pass

def unload_triposg_models():
    """卸载TripoSG模型以释放GPU内存"""
    global _triposg_pipe, _triposg_rmbg
    
    if _triposg_pipe is not None:
        try:
            # 清理管道
            if hasattr(_triposg_pipe, 'to'):
                _triposg_pipe.to('cpu')
            del _triposg_pipe
            _triposg_pipe = None
        except Exception as e:
            print(f"卸载TripoSG管道时出错: {e}")
    
    if _triposg_rmbg is not None:
        try:
            # 清理背景移除模型
            if hasattr(_triposg_rmbg, 'to'):
                _triposg_rmbg.to('cpu')
            del _triposg_rmbg
            _triposg_rmbg = None
        except Exception as e:
            print(f"卸载RMBG模型时出错: {e}")
    
    # 强制垃圾回收和CUDA清理
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return "✅ TripoSG模型已卸载"

def triposg_inference_ui(image, seed, steps, guidance, faces, num_tokens, use_flash_decoder, flash_octree_depth,auto_unload=True):
    """Gradio callback for TriposG - 修复版本"""
    try:
        import gradio as gr
        if image is None:
            return gr.update(), gr.update(), "请上传图片"

        # 验证输入参数
        if steps < 10 or steps > 100:
            return gr.update(), gr.update(), "推理步数应在10-100之间"
        if guidance < 1.0 or guidance > 15.0:
            return gr.update(), gr.update(), "Guidance Scale应在1.0-15.0之间"
        if num_tokens < 512 or num_tokens > 4096:
            return gr.update(), gr.update(), "Token数量应在512-4096之间"
        if flash_octree_depth < 6 or flash_octree_depth > 12:
            return gr.update(), gr.update(), "八叉树深度应在6-12之间"

        try:
            mesh = _run_triposg_inference(
                image_input=image,
                seed=int(seed) if seed != -1 else -1,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                num_tokens=int(num_tokens),
                use_flash_decoder=bool(use_flash_decoder),
                flash_octree_depth=int(flash_octree_depth),
                faces=int(faces)
            )
            
            if mesh is None:
                return gr.update(), gr.update(), "生成失败：未得到有效网格"


        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"TripoSG inference error: {error_details}")
            return gr.update(), gr.update(), f"模型推理失败: {str(e)}"

        # 导出为 GLB 文件
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
            mesh.export(tmp.name)
            glb_path = tmp.name
            
            # 验证文件是否成功创建
            if not os.path.exists(glb_path) or os.path.getsize(glb_path) == 0:
                return gr.update(), gr.update(), "导出GLB文件失败"
                
            file_size = os.path.getsize(glb_path) / 1024 / 1024 
            status_msg = f"推理完成 ✅ (种子: {seed}, 文件大小: {file_size:.2f}MB, 面数: {len(mesh.faces)})"
            if auto_unload:
                unload_status = unload_triposg_models()
                status_msg += f" | {unload_status}"
                return glb_path, glb_path, status_msg
            
        except Exception as export_error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return gr.update(), gr.update(), f"导出失败: {str(export_error)}"
            
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
def ordered_ui_categories():
    categories = [
        "prompt",
        "image", 
        "dimensions",
        "cfg",
        "checkboxes", 
        "accordions",
        "batch",
        "override_settings", 
        "scripts"
    ]
    
    # 用户自定义排序
    user_order = {}
    if hasattr(shared.opts, 'ui_reorder_list') and shared.opts.ui_reorder_list:
        user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}
    
    # 排序并返回
    for _, category in sorted(enumerate(categories), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category

# 添加缺失的符号定义
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol = '\u267b\ufe0f'  # ♻️
paste_symbol = '\u2199\ufe0f'  # ↙
refresh_symbol = '\U0001f504'  # 🔄
save_style_symbol = '\U0001f4be'  # 💾
apply_style_symbol = '\U0001f4cb'  # 📋
clear_prompt_symbol = '\U0001f5d1\ufe0f'  # 🗑️
extra_networks_symbol = '\U0001F3B4'  # 🎴
switch_values_symbol = '\U000021C5' # ⇅
restore_progress_symbol = '\U0001F300' # 🌀
detect_image_size_symbol = '\U0001F4D0'  # 📐

plaintext_to_html = ui_common.plaintext_to_html

# 其余的函数保持不变...
create_setting_component = ui_settings.create_setting_component

if shared.opts is not None:
    warnings.filterwarnings("default" if shared.opts.show_warnings else "ignore", category=UserWarning)
    warnings.filterwarnings("default" if shared.opts.show_gradio_deprecation_warnings else "ignore", category=gr.deprecation.GradioDeprecationWarning)

# this is a fix for Windows users. Without it, javascript files will be served with text/html content-type and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('application/javascript', '.mjs')

# Likewise, add explicit content-type header for certain missing image types
mimetypes.add_type('image/webp', '.webp')
mimetypes.add_type('image/avif', '.avif')

if not cmd_opts.share and not cmd_opts.listen:
    # fix gradio phoning home
    gradio.utils.version_check = lambda: None
    gradio.utils.get_local_ip_address = lambda: '127.0.0.1'

if cmd_opts.ngrok is not None:
    import modules.ngrok as ngrok
    print('ngrok authtoken detected, trying to connect...')
    ngrok.connect(
        cmd_opts.ngrok,
        cmd_opts.port if cmd_opts.port is not None else 7860,
        cmd_opts.ngrok_options
        )


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


sample_img2img = "assets/stable-samples/img2img/sketch-mountains-input.jpg"
sample_img2img = sample_img2img if os.path.exists(sample_img2img) else None

# 以下函数保持不变...
def send_gradio_gallery_to_image(x):
    if len(x) == 0:
        return None
    return image_from_url_text(x[0])

def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
    if not enable:
        return ""

    p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
    p.calculate_target_resolution()

    return f"from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"

def resize_from_to_html(width, height, scale_by):
    target_width = int(width * scale_by)
    target_height = int(height * scale_by)

    if not target_width or not target_height:
        return "no image selected"

    return f"resize: from <span class='resolution'>{width}x{height}</span> to <span class='resolution'>{target_width}x{target_height}</span>"


def process_interrogate(interrogation_function, mode, ii_input_dir, ii_output_dir, *ii_singles):
    if mode in {0, 1, 3, 4}:
        return [interrogation_function(ii_singles[mode]), None]
    elif mode == 2:
        return [interrogation_function(ii_singles[mode]["image"]), None]
    elif mode == 5:
        assert not shared.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
        images = shared.listfiles(ii_input_dir)
        print(f"Will process {len(images)} images.")
        if ii_output_dir != "":
            os.makedirs(ii_output_dir, exist_ok=True)
        else:
            ii_output_dir = ii_input_dir

        for image in images:
            img = Image.open(image)
            filename = os.path.basename(image)
            left, _ = os.path.splitext(filename)
            print(interrogation_function(img), file=open(os.path.join(ii_output_dir, f"{left}.txt"), 'a', encoding='utf-8'))

        return [gr.update(), None]


def interrogate(image):
    prompt = shared.interrogator.interrogate(image.convert("RGB"))
    return gr.update() if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = deepbooru.model.tag(image)
    return gr.update() if prompt is None else prompt


def connect_clear_prompt(button):
    """Given clear button, prompt, and token_counter objects, setup clear prompt button click event"""
    button.click(
        _js="clear_prompt",
        fn=None,
        inputs=[],
        outputs=[],
    )


def update_token_counter(text, steps, styles, *, is_positive=True):
    params = script_callbacks.BeforeTokenCounterParams(text, steps, styles, is_positive=is_positive)
    script_callbacks.before_token_counter_callback(params)
    text = params.prompt
    steps = params.steps
    styles = params.styles
    is_positive = params.is_positive

    if shared.opts.include_styles_into_token_counters:
        apply_styles = shared.prompt_styles.apply_styles_to_prompt if is_positive else shared.prompt_styles.apply_negative_styles_to_prompt
        text = apply_styles(text, styles)

    try:
        text, _ = extra_networks.parse_prompt(text)

        if is_positive:
            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        else:
            prompt_flat_list = [text]

        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    token_count, max_length = max([model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])
    return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"


def update_negative_prompt_token_counter(*args):
    return update_token_counter(*args, is_positive=False)


def setup_progressbar(*args, **kwargs):
    pass


def apply_setting(key, value):
    if value is None:
        return gr.update()

    if shared.cmd_opts.freeze_settings:
        return gr.update()

    # dont allow model to be swapped when model hash exists in prompt
    if key == "sd_model_checkpoint" and opts.disable_weights_auto_swap:
        return gr.update()

    if key == "sd_model_checkpoint":
        ckpt_info = sd_models.get_closet_checkpoint_match(value)

        if ckpt_info is not None:
            value = ckpt_info.title
        else:
            return gr.update()

    comp_args = opts.data_labels[key].component_args
    if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
        return

    valtype = type(opts.data_labels[key].default)
    oldval = opts.data.get(key, None)
    opts.data[key] = valtype(value) if valtype != type(None) else value
    if oldval != value and opts.data_labels[key].onchange is not None:
        opts.data_labels[key].onchange()

    opts.save(shared.config_filename)
    return getattr(opts, key)


def create_output_panel(tabname, outdir, toprow=None):
    return ui_common.create_output_panel(tabname, outdir, toprow)


def ordered_ui_categories():
    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(shared.opts.ui_reorder_list)}

    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


def create_override_settings_dropdown(tabname, row):
    dropdown = gr.Dropdown([], label="Override settings", visible=False, elem_id=f"{tabname}_override_settings", multiselect=True)

    dropdown.change(
        fn=lambda x: gr.Dropdown.update(visible=bool(x)),
        inputs=[dropdown],
        outputs=[dropdown],
    )

    return dropdown

def create_ui():
    import modules.img2img
    import modules.txt2img
    if shared.opts is None:
        import modules.initialize as initialize
        initialize.initialize() 

    reload_javascript()
    parameters_copypaste.reset()
    settings = ui_settings.UiSettings()
    settings.register_settings()
    scripts.scripts_current = scripts.scripts_txt2img
    scripts.scripts_txt2img.initialize_scripts(is_img2img=False)

    def generate_jewelry_prompt(user_request: str) -> Tuple[str, str, str]:
        if not user_request.strip():
            return "", "", "❌ 请输入首饰描述"
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {SILICON_CLOUD_API_KEY}"
            }
            payload = {
                "model": "deepseek-ai/DeepSeek-V2.5",
                "messages": [
                    {"role": "system", "content": JEWELRY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"请为以下需求生成提示词：{user_request}"}
                ],
                "temperature": 0.3,
                "max_tokens": 512,
                "response_format": {"type": "json_object"}  
            }
            
            print(f"正在调用 SiliconFlow API...")
            print(f"使用模型: DeepSeek-V2.5")
            
            response = requests.post(SILICON_CLOUD_API_URL, headers=headers, json=payload, timeout=30)
            print(f"API响应状态: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"API返回内容: {content}")
                try:
                    prompt_data = json.loads(content)
                    positive = prompt_data.get("positive", "").strip()
                    negative = prompt_data.get("negative", "").strip()
                    
                    if positive and negative:
                        return positive, negative, "✅ 提示词生成成功"
                    else:
                        return "", "", "❌ JSON 缺少 positive 或 negative 字段"
                        
                except json.JSONDecodeError as e:
                    return "", "", f"❌ 返回内容不是合法 JSON: {str(e)}"
                    
            else:
                error_msg = response.json().get("error", {}).get("message", response.text)
                return "", "", f"❌ API 调用失败 ({response.status_code}): {error_msg}"
                
        except requests.exceptions.Timeout:
            return "", "", "❌ API 调用超时（30秒）"
        except requests.exceptions.RequestException as e:
            return "", "", f"❌ 网络请求失败: {str(e)}"
        except Exception as e:
            return "", "", f"❌ 未知错误: {str(e)}"
        
    def generate_jewelry_prompt_fallback(user_request: str) -> Tuple[str, str, str]:
        """改进的备用提示词生成函数"""
        if not user_request.strip():
            return "", "", "❌ 请输入首饰描述"
        
        # 详细的翻译映射
        translation_map = {
            "彝族": "Yi ethnic minority style",
            "苗族": "Miao ethnic minority style", 
            "藏族": "Tibetan ethnic style",
            "蒙古族": "Mongolian ethnic style",
            "维吾尔族": "Uyghur ethnic style",
            "壮族": "Zhuang ethnic style",
            "戒指": "ring",
            "项链": "necklace", 
            "耳环": "earrings",
            "手镯": "bracelet",
            "手链": "bracelet",
            "胸针": "brooch",
            "银饰": "silver jewelry",
            "金饰": "gold jewelry",
            "玉石": "jade",
            "翡翠": "emerald",
            "珍珠": "pearl",
            "钻石": "diamond",
            "简约": "minimalist",
            "复古": "vintage",
            "现代": "modern",
            "传统": "traditional",
            "华丽": "luxurious",
            "精致": "exquisite",
            "民族风": "ethnic style",
            "中华": "Chinese",
            "少数民族": "ethnic minority",
            "风格": "style",
            "设计": "design"
        }
        
        # 翻译用户输入
        translated_request = user_request
        for chinese, english in translation_map.items():
            translated_request = translated_request.replace(chinese, english)
        
        # 生成专业提示词
        base_positive = f"professional jewelry design, {translated_request}, "
        base_positive += "highly detailed, clean white background, studio lighting, "
        base_positive += "perfect for 3D modeling, elegant design, wearable jewelry, "
        base_positive += "masterpiece, 8k, sharp focus, intricate details, "
        base_positive += "polished metal, gemstone inlays, cultural patterns, "
        base_positive += "isolated on white background"
        
        base_negative = "blurry, low quality, bad anatomy, extra fingers, "
        base_negative += "missing fingers, extra limbs, missing limbs, disfigured, "
        base_negative += "malformed, mutilated, poorly drawn hands, poorly drawn face, "
        base_negative += "mutation, deformed, ugly, bad proportions, "
        base_negative += "fused fingers, too many fingers, long neck, cartoon, "
        base_negative += "3d, render, graphic, text, signature, watermark, logo, "
        base_negative += "body out of frame, background clutter, multiple items, "
        base_negative += "complex patterns, human body, person, hands, "
        base_negative += "photograph, photo, realistic, hyperrealistic, "
        base_negative += "busy background, shadows, reflections"
        
        return base_positive, base_negative, "✅ 提示词已生成（使用备用方案）"
    
    def generate_prompt_wrapper(user_request: str):
        """包装函数，优先使用DeepSeek，失败时使用备用方案"""
        try:
            positive, negative, status = generate_jewelry_prompt(user_request)
            if "失败" in status or "错误" in status:
                # 如果DeepSeek失败，使用备用方案
                positive, negative, status = generate_jewelry_prompt_fallback(user_request)
            return positive, negative, status
        except Exception as e:
            return generate_jewelry_prompt_fallback(user_request)

    # === 定义 TripoSG 共享组件 ===
    t_image = gr.Image(type="pil", label="输入图片")
    t_seed = gr.Number(value=-1, label="随机种子(-1表示随机)")
    t_steps = gr.Slider(10, 100, value=50, step=1, label="推理步数")
    t_guidance = gr.Slider(1, 15, value=7.0, step=0.5, label="Guidance Scale")
    t_num_tokens = gr.Slider(512, 4096, value=2048, step=512, label="Token数量")
    t_use_flash_decoder = gr.Checkbox(value=True, label="使用快速解码器")
    t_flash_octree_depth = gr.Slider(6, 12, value=9, step=1, label="快速八叉树深度")
    t_faces = gr.Number(value=-1, label="目标面数 (<=0 不简化)")
    t_btn = gr.Button("生成模型")
    t_model3d = gr.Model3D(label="GLB 预览")
    t_file = gr.File(label="下载结果文件")
    t_status = gr.Textbox(label="状态", interactive=False)

        # === Hunyuan3D 共享组件 ===
    h_image = gr.Image(type="pil", label="输入图片（单视角）")

    # 共享参数
    h_seed = gr.Number(value=-1, label="随机种子(-1表示随机)")
    h_steps = gr.Slider(10, 100, value=40, step=1, label="推理步数")
    h_guidance = gr.Slider(1, 15, value=7.5, step=0.1, label="引导尺度")
    h_octree = gr.Slider(64, 512, value=384, step=8, label="分辨率")
    h_chunks = gr.Slider(1000, 500000, value=8000, step=1000, label="块数量")
    h_rembg = gr.Checkbox(value=True, label="移除背景")
    h_faces = gr.Number(value=0, label="目标面数 (<=0 不简化)")
    h_file_type = gr.Dropdown(
        choices=['glb', 'obj', 'ply', 'stl'],
        value='glb',
        label="导出格式"
    )   
    h_btn = gr.Button("生成模型")
    h_model3d = gr.Model3D(label="GLB 预览")
    h_file = gr.File(label="下载结果文件")
    h_status = gr.Textbox(label="状态", interactive=False)

    def send_to_triposg(imgs):
        if not imgs:
            return gr.update(value=None), "❌ 无图片"
        try:
            img = image_from_url_text(imgs[0]) 
            if img is None:
                return gr.update(value=None), "❌ 无法加载图片"
            return gr.update(value=img), "✅ 已发送"
        except Exception as e:
            return gr.update(value=None), f"❌ 错误: {e}"
        
    def send_to_hunyuan(imgs):
        if not imgs:
            return gr.update(value=None), "❌ 无图片"
        try:
            img = image_from_url_text(imgs[0]) 
            if img is None:
                return gr.update(value=None), "❌ 无法加载图片"
            return gr.update(value=img), "✅ 已发送到 Hunyuan3D"
        except Exception as e:
            return gr.update(value=None), f"❌ 错误: {e}"

    # === txt2img Interface ===
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        toprow = ui_toprow.Toprow(is_img2img=False, is_compact=shared.opts.compact_prompt_box)
        dummy_component = gr.Label(visible=False)

        prompt_status_display = gr.Textbox(
            label="生成状态",
            interactive=False,
            visible=False,
            elem_id="prompt_status_display"
        )

        with gr.Accordion("🧠 DeepSeek 首饰提示词生成", open=False, elem_id="deepseek_prompt_generator"):
            with FormRow():
                jewelry_description = gr.Textbox(
                    label="首饰描述",
                    placeholder="描述您想要的首饰，例如：一枚简约的银色戒指，带有小钻石装饰",
                    lines=2,
                    elem_id="jewelry_description"
                )
            
            with FormRow():
                generate_prompt_btn = gr.Button("生成提示词", variant="primary", elem_id="generate_prompt_btn")
                prompt_status = gr.Textbox(label="状态", interactive=False, show_label=False)
            
            with FormRow():
                apply_prompt_btn = gr.Button("应用提示词", variant="secondary", elem_id="apply_prompt_btn")


        extra_tabs = gr.Tabs(elem_id="txt2img_extra_tabs", elem_classes=["extra-networks"])
        extra_tabs.__enter__()

        with gr.Tab("Generation", id="txt2img_generation") as txt2img_generation_tab:
            with ResizeHandleRow(equal_height=False):
                # Left: Settings
                with gr.Column(variant='compact', elem_id="txt2img_settings", scale=3):
                    with ExitStack() as stack:
                        if shared.opts.txt2img_settings_accordion:
                            stack.enter_context(gr.Accordion("Open for Settings", open=False))

                        scripts.scripts_txt2img.prepare_ui()

                        for category in ordered_ui_categories():
                            if category == "prompt":
                                toprow.create_inline_toprow_prompts()

                            elif category == "dimensions":
                                with FormRow():
                                    with gr.Column(elem_id="txt2img_column_size", scale=4):
                                        width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=1024, elem_id="txt2img_width")
                                        height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=1024, elem_id="txt2img_height")

                                    with gr.Column(elem_id="txt2img_dimensions_row", scale=1, elem_classes="dimensions-tools"):
                                        res_switch_btn = ToolButton(value=switch_values_symbol, elem_id="txt2img_res_switch_btn", tooltip="Switch width/height")

                                    if opts.dimensions_and_batch_together:
                                        with gr.Column(elem_id="txt2img_column_batch"):
                                            batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                            batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                            elif category == "cfg":
                                with gr.Row():
                                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="txt2img_cfg_scale")

                            elif category == "checkboxes":
                                with FormRow(elem_classes="checkboxes-row", variant="compact"):
                                    pass

                            elif category == "accordions":
                                with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
                                    with InputAccordion(False, label="Hires. fix", elem_id="txt2img_hr") as enable_hr:
                                        with enable_hr.extra():
                                            hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False, min_width=0)

                                        with FormRow(elem_id="txt2img_hires_fix_row1", variant="compact"):
                                            hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="txt2img_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                                            hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="txt2img_hires_steps")
                                            denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="txt2img_denoising_strength")

                                        with FormRow(elem_id="txt2img_hires_fix_row2", variant="compact"):
                                            hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="txt2img_hr_scale")
                                            hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=0, elem_id="txt2img_hr_resize_x")
                                            hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=0, elem_id="txt2img_hr_resize_y")

                                        with FormRow(elem_id="txt2img_hires_fix_row3", variant="compact", visible=opts.hires_fix_show_sampler) as hr_sampler_container:
                                            hr_checkpoint_name = gr.Dropdown(label='Checkpoint', elem_id="hr_checkpoint", choices=["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint")
                                            create_refresh_button(hr_checkpoint_name, modules.sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + modules.sd_models.checkpoint_tiles(use_short=True)}, "hr_checkpoint_refresh")

                                            hr_sampler_name = gr.Dropdown(label='Hires sampling method', elem_id="hr_sampler", choices=["Use same sampler"] + sd_samplers.visible_sampler_names(), value="Use same sampler")
                                            hr_scheduler = gr.Dropdown(label='Hires schedule type', elem_id="hr_scheduler", choices=["Use same scheduler"] + [x.label for x in sd_schedulers.schedulers], value="Use same scheduler")

                                        with FormRow(elem_id="txt2img_hires_fix_row4", variant="compact", visible=opts.hires_fix_show_prompts) as hr_prompts_container:
                                            with gr.Column(scale=80):
                                                with gr.Row():
                                                    hr_prompt = gr.Textbox(label="Hires prompt", elem_id="hires_prompt", show_label=False, lines=3, placeholder="Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.", elem_classes=["prompt"])
                                            with gr.Column(scale=80):
                                                with gr.Row():
                                                    hr_negative_prompt = gr.Textbox(label="Hires negative prompt", elem_id="hires_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.", elem_classes=["prompt"])

                                    scripts.scripts_txt2img.setup_ui_for_section(category)

                            elif category == "batch":
                                if not opts.dimensions_and_batch_together:
                                    with FormRow(elem_id="txt2img_column_batch"):
                                        batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="txt2img_batch_count")
                                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="txt2img_batch_size")

                            elif category == "override_settings":
                                with FormRow(elem_id="txt2img_override_settings_row") as row:
                                    override_settings = create_override_settings_dropdown('txt2img', row)

                            elif category == "scripts":
                                with FormGroup(elem_id="txt2img_script_container"):
                                    custom_inputs = scripts.scripts_txt2img.setup_ui()

                            if category not in {"accordions"}:
                                scripts.scripts_txt2img.setup_ui_for_section(category)

                    hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
                    for component in hr_resolution_preview_inputs:
                        event = component.release if isinstance(component, gr.Slider) else component.change
                        event(
                            fn=calc_resolution_hires,
                            inputs=hr_resolution_preview_inputs,
                            outputs=[hr_final_resolution],
                            show_progress=False,
                        )
                        event(
                            None,
                            _js="onCalcResolutionHires",
                            inputs=hr_resolution_preview_inputs,
                            outputs=[],
                            show_progress=False,
                        )

                # Right: Output + Send Button
                with gr.Column(scale=4):
                    output_panel = create_output_panel("txt2img", opts.outdir_txt2img_samples, toprow)

                    output_panel.infotext.visible = False
                    output_panel.html_log.visible = False

                    with gr.Row():
                        send_to_triposg_btn = gr.Button("📤 发送到 TripoSG", elem_id="txt2img_send_to_triposg")
                        send_to_hunyuan_btn = gr.Button("📤 发送到 Hunyuan3D", elem_id="txt2img_send_to_hunyuan")

                    send_status = gr.Textbox(label="发送状态", visible=False)

                    send_to_triposg_btn.click(
                        fn=send_to_triposg,
                        inputs=[output_panel.gallery],
                        outputs=[t_image, send_status],
                    )
                    send_to_hunyuan_btn.click(
                        fn=send_to_hunyuan,
                        inputs=[output_panel.gallery],
                        outputs=[h_image, send_status],
                    )
            txt2img_inputs = [
                dummy_component,
                toprow.prompt,
                toprow.negative_prompt,
                toprow.ui_styles.dropdown,
                batch_count,
                batch_size,
                cfg_scale,
                height,
                width,
                enable_hr,
                denoising_strength,
                hr_scale,
                hr_upscaler,
                hr_second_pass_steps,
                hr_resize_x,
                hr_resize_y,
                hr_checkpoint_name,
                hr_sampler_name,
                hr_scheduler,
                hr_prompt,
                hr_negative_prompt,
                override_settings,
            ] + custom_inputs

            txt2img_outputs = [
                output_panel.gallery,
                output_panel.generation_info,
                output_panel.infotext,
                output_panel.html_log,
            ]

            txt2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                _js="submit",
                inputs=txt2img_inputs,
                outputs=txt2img_outputs,
                show_progress=False,
            )

            toprow.prompt.submit(**txt2img_args)
            toprow.submit.click(**txt2img_args)

            output_panel.button_upscale.click(
                fn=wrap_gradio_gpu_call(modules.txt2img.txt2img_upscale, extra_outputs=[None, '', '']),
                _js="submit_txt2img_upscale",
                inputs=txt2img_inputs[0:1] + [output_panel.gallery, dummy_component, output_panel.generation_info] + txt2img_inputs[1:],
                outputs=txt2img_outputs,
                show_progress=False,
            )

            res_switch_btn.click(fn=None, _js="function(){switchWidthHeight('txt2img')}", inputs=None, outputs=None, show_progress=False)

            toprow.restore_progress_button.click(
                fn=progress.restore_progress,
                _js="restoreProgressTxt2img",
                inputs=[dummy_component],
                outputs=txt2img_outputs,
                show_progress=False,
            )

            txt2img_paste_fields = [
                PasteField(toprow.prompt, "Prompt", api="prompt"),
                PasteField(toprow.negative_prompt, "Negative prompt", api="negative_prompt"),
                PasteField(cfg_scale, "CFG scale", api="cfg_scale"),
                PasteField(width, "Size-1", api="width"),
                PasteField(height, "Size-2", api="height"),
                PasteField(batch_size, "Batch size", api="batch_size"),
                PasteField(toprow.ui_styles.dropdown, lambda d: d["Styles array"] if isinstance(d.get("Styles array"), list) else gr.update(), api="styles"),
                PasteField(denoising_strength, "Denoising strength", api="denoising_strength"),
                PasteField(enable_hr, lambda d: "Denoising strength" in d and ("Hires upscale" in d or "Hires upscaler" in d or "Hires resize-1" in d), api="enable_hr"),
                PasteField(hr_scale, "Hires upscale", api="hr_scale"),
                PasteField(hr_upscaler, "Hires upscaler", api="hr_upscaler"),
                PasteField(hr_second_pass_steps, "Hires steps", api="hr_second_pass_steps"),
                PasteField(hr_resize_x, "Hires resize-1", api="hr_resize_x"),
                PasteField(hr_resize_y, "Hires resize-2", api="hr_resize_y"),
                PasteField(hr_checkpoint_name, "Hires checkpoint", api="hr_checkpoint_name"),
                PasteField(hr_sampler_name, sd_samplers.get_hr_sampler_from_infotext, api="hr_sampler_name"),
                PasteField(hr_scheduler, sd_samplers.get_hr_scheduler_from_infotext, api="hr_scheduler"),
                PasteField(hr_sampler_container, lambda d: gr.update(visible=True) if d.get("Hires sampler", "Use same sampler") != "Use same sampler" or d.get("Hires checkpoint", "Use same checkpoint") != "Use same checkpoint" or d.get("Hires schedule type", "Use same scheduler") != "Use same scheduler" else gr.update()),
                PasteField(hr_prompt, "Hires prompt", api="hr_prompt"),
                PasteField(hr_negative_prompt, "Hires negative prompt", api="hr_negative_prompt"),
                PasteField(hr_prompts_container, lambda d: gr.update(visible=True) if d.get("Hires prompt", "") != "" or d.get("Hires negative prompt", "") != "" else gr.update()),
                *scripts.scripts_txt2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("txt2img", None, txt2img_paste_fields, override_settings)
            parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                paste_button=toprow.paste, tabname="txt2img", source_text_component=toprow.prompt, source_image_component=None,
            ))

            generated_positive = gr.State("")
            generated_negative = gr.State("")
            
            generate_prompt_btn.click(
                fn=generate_prompt_wrapper,
                inputs=[jewelry_description],
                outputs=[generated_positive, generated_negative, prompt_status]
            )
            
            apply_prompt_btn.click(
                fn=lambda pos, neg: (pos, neg, "✅ 提示词已应用"),
                inputs=[generated_positive, generated_negative],
                outputs=[toprow.prompt, toprow.negative_prompt, prompt_status]
            )

            prompt_status.change(
                fn=lambda x: gr.update(visible=True, value=x) if x else gr.update(visible=False),
                inputs=[prompt_status],
                outputs=[prompt_status_display]
            )

            steps = scripts.scripts_txt2img.script('Sampler').steps
            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.ui_styles.dropdown.change(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])
            toprow.token_button.click(fn=wrap_queued_call(update_token_counter), inputs=[toprow.prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.token_counter])
            toprow.negative_token_button.click(fn=wrap_queued_call(update_negative_prompt_token_counter), inputs=[toprow.negative_prompt, steps, toprow.ui_styles.dropdown], outputs=[toprow.negative_token_counter])

        extra_networks_ui = ui_extra_networks.create_ui(txt2img_interface, [txt2img_generation_tab], 'txt2img')
        ui_extra_networks.setup_ui(extra_networks_ui, output_panel.gallery)
        extra_tabs.__exit__()
    
    
    

    # === TripoSG Interface ===
    with gr.Blocks(analytics_enabled=False) as triposg_interface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🟢 TripoSG")
                t_image.render()
                with gr.Accordion("扩散参数", open=True):
                    t_seed.render()
                    t_steps.render()
                    t_guidance.render()
                    t_num_tokens.render()
                with gr.Accordion("几何提取参数", open=True):
                    t_use_flash_decoder.render()
                    t_flash_octree_depth.render()
                    t_faces.render()
                t_btn.render()
            with gr.Column():
                gr.Markdown("### 预览与下载")
                t_model3d.render()
                t_file.render()
                t_status.render()

        t_btn.click(
            fn=wrap_gradio_gpu_call(triposg_inference_ui),
            inputs=[t_image, t_seed, t_steps, t_guidance, t_faces, t_num_tokens, t_use_flash_decoder, t_flash_octree_depth],
            outputs=[t_model3d, t_file, t_status]
        )
    # === Hunyuan3D Interface ===
    with gr.Blocks(analytics_enabled=False) as hunyuan_interface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔵 Hunyuan3D")
                h_image.render()
                h_seed.render()
                h_steps.render()
                h_guidance.render()
                h_octree.render()
                h_chunks.render()
                with gr.Accordion("高级选项", open=False):
                    h_rembg.render()
                h_btn.render()
            with gr.Column():
                gr.Markdown("### 预览与下载")
                h_model3d.render()
                h_file.render()
                h_status.render()
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 导出选项")
                with gr.Row():
                    h_export_faces = gr.Number(value=0, label="目标面数 (<=0 不简化)")
                    h_export_format = gr.Dropdown(
                        choices=['glb', 'obj', 'ply', 'stl'],
                        value='glb',
                        label="导出格式"
                    )
                h_export_btn = gr.Button("导出模型", variant="secondary")
            with gr.Column():
                h_export_file = gr.File(label="导出文件")
                h_export_status = gr.Textbox(label="导出状态", interactive=False)
        h_btn.click(
            fn=wrap_gradio_gpu_call(hunyuan_processor.generate_3d),
            inputs=[h_image, h_steps, h_guidance, h_seed, h_octree, h_rembg, h_chunks],
            outputs=[h_model3d, h_file, h_status]
        )
        h_export_btn.click(
            fn=wrap_gradio_gpu_call(hunyuan_processor.export_mesh),
            inputs=[h_file, h_export_faces, h_export_format],
            outputs=[h_export_file, h_export_status]
        )
    # === Rest of UI setup ===
    loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
    ui_settings_from_file = loadsave.ui_settings.copy()
    settings.create_ui(loadsave, dummy_component)

    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (triposg_interface, "TripoSG", "triposg"),
        (hunyuan_interface, "Hunyuan3D", "hunyuan3d"),
    ]

    interfaces += script_callbacks.ui_tabs_callback()
    extensions_interface = ui_extensions.create_ui()\
    

    interfaces += [(extensions_interface, "Extensions", "extensions")]
    interfaces += [(settings.interface, "Settings", "settings")]


    shared.tab_names = []
    for _interface, label, _ifid in interfaces:
        shared.tab_names.append(label)

    with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="Stable Diffusion") as demo:
        settings.add_quicksettings()
        parameters_copypaste.connect_paste_params_buttons()

        with gr.Tabs(elem_id="tabs") as tabs:
            tab_order = {k: i for i, k in enumerate(opts.ui_tab_order)}
            sorted_interfaces = sorted(interfaces, key=lambda x: tab_order.get(x[1], 9999))

            for interface, label, ifid in sorted_interfaces:
                if label in shared.opts.hidden_tabs:
                    continue
                with gr.TabItem(label, id=ifid, elem_id=f"tab_{ifid}"):
                    interface.render()

                if ifid not in ["extensions", "settings"]:
                    loadsave.add_block(interface, ifid)

            loadsave.add_component(f"webui/Tabs@{tabs.elem_id}", tabs)
            loadsave.setup_ui()

        if os.path.exists(os.path.join(script_path, "notification.mp3")) and shared.opts.notification_audio:
            gr.Audio(interactive=False, value=os.path.join(script_path, "notification.mp3"), elem_id="audio_notification", visible=False)

        footer = shared.html("footer.html")
        footer = footer.format(versions=versions_html(), api_docs="/docs" if shared.cmd_opts.api else "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API")
        gr.HTML(footer, elem_id="footer")
        settings.add_functionality(demo)

    if ui_settings_from_file != loadsave.ui_settings:
        loadsave.dump_defaults()
    demo.ui_loadsave = loadsave

    return demo

def versions_html():
    import torch
    import launch

    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    commit = launch.commit_hash()
    tag = launch.git_tag()

    if shared.xformers_available:
        import xformers
        xformers_version = xformers.__version__
    else:
        xformers_version = "N/A"

    return f"""
version: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/{commit}">{tag}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{python_version}</span>
&#x2000;•&#x2000;
torch: {getattr(torch, '__long_version__',torch.__version__)}
&#x2000;•&#x2000;
xformers: {xformers_version}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
"""


def setup_ui_api(app):
    from pydantic import BaseModel, Field

    class QuicksettingsHint(BaseModel):
        name: str = Field(title="Name of the quicksettings field")
        label: str = Field(title="Label of the quicksettings field")

    def quicksettings_hint():
        return [QuicksettingsHint(name=k, label=v.label) for k, v in opts.data_labels.items()]

    app.add_api_route("/internal/quicksettings-hint", quicksettings_hint, methods=["GET"], response_model=list[QuicksettingsHint])

    app.add_api_route("/internal/ping", lambda: {}, methods=["GET"])

    app.add_api_route("/internal/profile-startup", lambda: timer.startup_record, methods=["GET"])

    def download_sysinfo(attachment=False):
        from fastapi.responses import PlainTextResponse

        text = sysinfo.get()
        filename = f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.json"

        return PlainTextResponse(text, headers={'Content-Disposition': f'{"attachment" if attachment else "inline"}; filename="{filename}"'})

    app.add_api_route("/internal/sysinfo", download_sysinfo, methods=["GET"])
    app.add_api_route("/internal/sysinfo-download", lambda: download_sysinfo(attachment=True), methods=["GET"])

    import fastapi.staticfiles
    app.mount("/webui-assets", fastapi.staticfiles.StaticFiles(directory=launch_utils.repo_dir('stable-diffusion-webui-assets')), name="webui-assets")