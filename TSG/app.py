import argparse
import os
import sys
import tempfile
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from briarmbg import BriaRMBG
import pymeshlab
import gradio as gr

# 模型权重目录
TRI_POSG_WEIGHTS_DIR = "D:\work\AUTO1111\webui\TripoSG\pretrained_weights\TripoSG"
RMBG_WEIGHTS_DIR = "D:\work\AUTO1111\webui\TripoSG\pretrained_weights\RMBG-1.4"

# ========== 原始推理逻辑保留 ==========
@torch.no_grad()
def run_triposg(
    pipe,
    image_input,
    rmbg_net,
    seed: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    faces: int = -1,
) -> trimesh.Trimesh:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]

    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)

    return mesh

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    return trimesh.Trimesh(vertices=verts, faces=faces)

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh

# ========== 初始化模型 ==========
device = "cuda"
dtype = torch.float16

rmbg_net = BriaRMBG.from_pretrained(RMBG_WEIGHTS_DIR).to(device)
rmbg_net.eval()
pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(TRI_POSG_WEIGHTS_DIR).to(device, dtype)

# ========== Gradio 包装 ==========
def inference_ui(image, seed, steps, guidance, faces):
    mesh = run_triposg(
        pipe,
        image_input=image,
        rmbg_net=rmbg_net,
        seed=int(seed),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        faces=int(faces),
    )

    # 临时文件写入 GLB
    tmp_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    mesh.export(tmp_file.name)
    tmp_file.close()

    # 返回文件路径给 Model3D 和 File
    return tmp_file.name, tmp_file.name

with gr.Blocks() as demo:
    gr.Markdown("## 🟢 TripoSG ")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="输入图片")
            seed = gr.Number(value=42, label="随机种子")
            steps = gr.Slider(10, 100, value=50, step=1, label="推理步数")
            guidance = gr.Slider(1, 15, value=7.0, step=0.5, label="Guidance Scale")
            faces = gr.Number(value=-1, label="目标面数 (<=0 不简化)")
            btn = gr.Button("生成模型")
        with gr.Column():
            gr.Markdown("### 预览与下载")
            model3d = gr.Model3D(label="GLB 预览")
            download = gr.File(label="下载 GLB")

    btn.click(
        fn=inference_ui,
        inputs=[image, seed, steps, guidance, faces],
        outputs=[model3d, download],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
