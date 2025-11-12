# modules/hunyuan3d_ui.py
import os
import sys
import torch
import trimesh
from pathlib import Path
from PIL import Image
import uuid
import tempfile
import torch.nn as nn
_original_linear_forward = nn.Linear.forward
_TRIPOSG_WORKDIR = r"D:\work\AUTO1111\webui\TripoSG"
_RMBG_WEIGHTS_DIR = os.path.join(_TRIPOSG_WORKDIR, "pretrained_weights", "RMBG-1.4")
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hunyuan_dir = os.path.join(current_dir, "Hunyuan3D-2.1")
sys.path.insert(0, hunyuan_dir)
sys.path.insert(0, os.path.join(hunyuan_dir, "hy3dshape"))
sys.path.insert(0, os.path.join(hunyuan_dir, "hy3dpaint"))

try:
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover
except ImportError as e:
    raise ImportError(f"❌ Hunyuan3D 模块导入失败: {e}")


def simplify_mesh(mesh: trimesh.Trimesh, n_faces: int):
    if n_faces <= 0 or mesh.faces.shape[0] <= n_faces:
        return mesh
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    ms.meshing_merge_close_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
    simplified = ms.current_mesh()
    return trimesh.Trimesh(vertices=simplified.vertex_matrix(), faces=simplified.face_matrix())

class Hunyuan3DProcessor:
    def __init__(self):
        self.pipeline = None
        self.floater_remover = None
        self.rmbg = None
        self.degenerate_remover = None   
        self.face_reducer = None 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(hunyuan_dir, "model", "Hunyuan3D-2.1")
        self.subfolder = "hunyuan3d-dit-v2-1"
        self.low_vram_mode = True

    def load_models(self):
        if self.pipeline is not None:
            return
        original_apply = None
        try:
            from extensions_builtin.Lora import lora
            original_apply = lora.lora_apply_weights
            lora.lora_apply_weights = lambda: None
        except:
            pass

        model_full_path = os.path.join(self.model_path, self.subfolder)
        if not os.path.exists(os.path.join(model_full_path, "config.yaml")):
            raise FileNotFoundError(f"模型路径不存在或缺少 config.yaml: {model_full_path}")

        try:
            print(f"正在加载 Hunyuan3D 模型...")
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_path,
                subfolder=self.subfolder,
                device=self.device,
                torch_dtype=torch.float16,
                use_safetensors=False,
            )
            if self.rmbg is None:
                self.rmbg = BackgroundRemover()
            if self.floater_remover is None:
                self.floater_remover = FloaterRemover()
            if self.degenerate_remover is None:
                self.degenerate_remover = DegenerateFaceRemover()
            if self.face_reducer is None:
                self.face_reducer = FaceReducer()
            print("✅ Hunyuan3D 模型加载完成")
        finally:
            if original_apply is not None:
                from extensions_builtin.Lora import lora
                lora.lora_apply_weights = original_apply

    def unload_model(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @torch.no_grad()
    def generate_3d(self, 
                    image=None,
                    steps=40, 
                    guidance_scale=7.5, 
                    seed=-1,
                    octree_resolution=384, 
                    check_box_rembg=True, 
                    num_chunks=8000):
        try:
            self.load_models()
            
            if image is None:
                raise ValueError("必须提供 image")
            if check_box_rembg and self.rmbg is not None:
                if image.mode == 'RGBA':
                    image = self.rmbg(image.convert('RGB'))
                elif image.mode == 'RGB':
                    image = self.rmbg(image)
            input_image = image
            main_preview = image

            generator = torch.Generator(device=self.device)
            if seed != -1:
                generator = generator.manual_seed(int(seed))

            original_linear_forward = nn.Linear.forward
            try:
                nn.Linear.forward = _original_linear_forward 
                outputs = self.pipeline(
                    image=input_image,
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    generator=generator,
                    octree_resolution=int(octree_resolution),
                    num_chunks=int(num_chunks),
                    output_type='mesh'
                )
            finally:
                nn.Linear.forward = original_linear_forward

            mesh = export_to_trimesh(outputs)[0]
            save_dir = os.path.join(current_dir, "outputs", "hunyuan3d")
            os.makedirs(save_dir, exist_ok=True)
            glb_path = os.path.join(save_dir, f"{uuid.uuid4()}.glb")
            try:
                mesh.export(glb_path)
            except Exception as e:
                raise RuntimeError(f"导出失败: {e}")

            if not os.path.exists(glb_path) or os.path.getsize(glb_path) == 0:
                raise RuntimeError("GLB 文件未正确生成")

            return glb_path, glb_path, f"✅ 生成完成\n面数: {mesh.faces.shape[0]}\n顶点数: {mesh.vertices.shape[0]}"
        finally:
        #     if self.low_vram_mode:
        #         self.unload_model()
            pass
    
    @torch.no_grad()
    def export_mesh(self, file_component_value, target_faces: int = 0, file_type: str = 'glb'):
        """对已有 .glb 模型进行后处理并导出为指定格式"""
        if isinstance(file_component_value, str):
            glb_path = file_component_value
        elif isinstance(file_component_value, dict):
            glb_path = file_component_value.get("name", "")
        elif hasattr(file_component_value, 'name') and isinstance(file_component_value.name, str):
            glb_path = file_component_value.name
        else:
            raise ValueError(f"❌ 无法识别的文件输入类型: {type(file_component_value)}")

        if not glb_path or not os.path.exists(glb_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {glb_path}")
        
        target_faces = int(target_faces)

        mesh = trimesh.load(glb_path)
        if self.floater_remover is not None:
            mesh = self.floater_remover(mesh)
        if self.degenerate_remover is not None:
            mesh = self.degenerate_remover(mesh)
        if target_faces > 0 and self.face_reducer is not None:
            mesh = self.face_reducer(mesh, max_facenum=target_faces)

        save_dir = os.path.dirname(glb_path)
        ext = file_type.lower().lstrip('.')
        if ext not in ['glb', 'obj', 'ply', 'stl']:
            ext = 'glb'
        output_path = os.path.join(save_dir, f"exported_{uuid.uuid4()}.{ext}")
        mesh.export(output_path)
        return output_path, f"✅ 导出完成 ({os.path.basename(output_path)})"

# 全局实例
hunyuan_processor = Hunyuan3DProcessor()