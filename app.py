# === app.py (نسخة معدلة بالكامل) ===
import os
import io
import base64
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
import torch
import numpy as np

from pygltflib import GLTF2, Scene, Node, Mesh as GLTFMesh, Buffer, BufferView, Accessor, Asset, Primitive, Image as GLTFImage, Texture, Material, TextureInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hunyuan3D-2 API",
    description="High-Resolution 3D Assets Generation API (GLB with Texture)",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = None

class GenerationRequest(BaseModel):
    image_base64: str
    quality: str = "high"
    output_format: str = "glb"

class GenerationResponse(BaseModel):
    success: bool
    message: str
    model_url: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

async def load_model():
    global model, device
    try:
        logger.info("Loading Hunyuan3D-2 model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from hy3dgen import Hunyuan3DPipeline
        model = Hunyuan3DPipeline.from_pretrained(
            "Tencent-Hunyuan/Hunyuan3D-2",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        if device.type == "cuda":
            model = model.to(device)
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    success = await load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.post("/generate", response_model=GenerationResponse)
async def generate_3d_model(
    image: UploadFile = File(...),
    quality: str = Form("high"),
    output_format: str = Form("glb")
):
    global model, device
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")

        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input.png")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        pil_image = Image.open(input_path).convert("RGB")

        quality_settings = {
            "high": {"steps": 50, "guidance_scale": 7.5},
            "medium": {"steps": 30, "guidance_scale": 7.0},
            "low": {"steps": 20, "guidance_scale": 6.5}
        }
        settings = quality_settings.get(quality, quality_settings["high"])

        logger.info(f"Generating 3D mesh with texture for quality: {quality}")
        with torch.no_grad():
            result = model(
                image=pil_image,
                num_inference_steps=settings["steps"],
                guidance_scale=settings["guidance_scale"],
                output_type="mesh",
                return_dict=True
            )

        mesh = result.meshes[0]
        texture_image = pil_image

        model_filename = f"model.glb"
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, model_filename)

        save_glb_with_texture(mesh, texture_image, model_path)

        preview_path = os.path.join(output_dir, "preview.png")
        generate_preview(mesh, preview_path)

        permanent_dir = "/tmp/outputs"
        os.makedirs(permanent_dir, exist_ok=True)
        import uuid
        unique_id = str(uuid.uuid4())

        final_model_path = os.path.join(permanent_dir, f"{unique_id}_model.glb")
        final_preview_path = os.path.join(permanent_dir, f"{unique_id}_preview.png")

        shutil.move(model_path, final_model_path)
        shutil.move(preview_path, final_preview_path)
        shutil.rmtree(temp_dir)

        response_data = {
            "success": True,
            "message": "3D GLB with texture generated",
            "model_url": f"/download/{unique_id}_model.glb",
            "preview_url": f"/download/{unique_id}_preview.png",
            "metadata": {
                "format": "glb",
                "quality": quality
            }
        }
        return GenerationResponse(**response_data)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed: {e}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("/tmp/outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

def save_glb_with_texture(mesh, texture_image: Image.Image, path: str):
    """
    Save mesh + texture as proper GLB using pygltflib.
    """
    vertices = mesh.verts_list()[0].cpu().numpy().astype(np.float32)
    faces = mesh.faces_list()[0].cpu().numpy().astype(np.uint32)

    vertex_bytes = vertices.tobytes()
    index_bytes = faces.flatten().tobytes()

    tex_bytes_io = io.BytesIO()
    texture_image.save(tex_bytes_io, format='PNG')
    tex_bytes = tex_bytes_io.getvalue()

    buffer_data = vertex_bytes + index_bytes + tex_bytes

    buffer = Buffer(byteLength=len(buffer_data))
    buffer_view_pos = BufferView(buffer=0, byteOffset=0, byteLength=len(vertex_bytes))
    buffer_view_idx = BufferView(buffer=0, byteOffset=len(vertex_bytes), byteLength=len(index_bytes))
    buffer_view_img = BufferView(buffer=0, byteOffset=len(vertex_bytes) + len(index_bytes), byteLength=len(tex_bytes))

    accessor_pos = Accessor(bufferView=0, componentType=5126, count=len(vertices), type="VEC3")
    accessor_idx = Accessor(bufferView=1, componentType=5125, count=len(faces.flatten()), type="SCALAR")

    gltf_img = GLTFImage(bufferView=2, mimeType="image/png")
    texture = Texture(source=0)
    material = Material(pbrMetallicRoughness=None, name="RoomMaterial", doubleSided=True,
                        baseColorTexture=TextureInfo(index=0))

    primitive = Primitive(attributes={"POSITION": 0}, indices=1, material=0)
    mesh_gltf = GLTFMesh(primitives=[primitive])

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        buffers=[buffer],
        bufferViews=[buffer_view_pos, buffer_view_idx, buffer_view_img],
        accessors=[accessor_pos, accessor_idx],
        images=[gltf_img],
        textures=[texture],
        materials=[material],
        meshes=[mesh_gltf],
        nodes=[Node(mesh=0)],
        scenes=[Scene(nodes=[0])],
        scene=0
    )
    gltf.set_binary_blob(buffer_data)
    gltf.save_binary(path)

def generate_preview(mesh, path: str):
    try:
        import matplotlib.pyplot as plt
        vertices = mesh.verts_list()[0].cpu().numpy()
        faces = mesh.faces_list()[0].cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                        triangles=faces, cmap='viridis')
        plt.savefig(path)
        plt.close()
    except Exception as e:
        logger.warning(f"Preview failed: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )
