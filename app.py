import os
import io
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hunyuan3D-2 API",
    description="High-Resolution 3D Assets Generation API",
    version="1.0.0"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
device = None

class GenerationRequest(BaseModel):
    image_base64: str
    quality: str = "high"  # high, medium, low
    colored: bool = True
    output_format: str = "obj"  # obj, ply, gltf

class GenerationResponse(BaseModel):
    success: bool
    message: str
    model_url: Optional[str] = None
    texture_url: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

async def load_model():
    """Load the Hunyuan3D-2 model"""
    global model, device
    try:
        logger.info("Loading Hunyuan3D-2 model...")
        
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Import Hunyuan3D modules
        from hy3dgen import Hunyuan3DPipeline
        
        # Initialize the pipeline
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
    """Initialize model on startup"""
    success = await load_model()
    if not success:
        logger.error("Failed to load model on startup")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Hunyuan3D-2 API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_3d_model(
    image: UploadFile = File(...),
    quality: str = Form("high"),
    colored: bool = Form(True),
    output_format: str = Form("obj")
):
    """Generate 3D model from 2D image"""
    global model, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate input image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, f"input.{image.filename.split('.')[-1]}")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save uploaded image
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Load and process image
        pil_image = Image.open(input_path).convert("RGB")
        
        # Set generation parameters based on quality
        quality_settings = {
            "high": {"resolution": 1024, "steps": 50, "guidance_scale": 7.5},
            "medium": {"resolution": 512, "steps": 30, "guidance_scale": 7.0},
            "low": {"resolution": 256, "steps": 20, "guidance_scale": 6.5}
        }
        
        settings = quality_settings.get(quality, quality_settings["high"])
        
        logger.info(f"Generating 3D model with quality: {quality}")
        
        # Generate 3D model
        with torch.no_grad():
            result = model(
                image=pil_image,
                num_inference_steps=settings["steps"],
                guidance_scale=settings["guidance_scale"],
                output_type="mesh",
                return_dict=True
            )
        
        # Extract mesh data
        mesh = result.meshes[0]
        
        # Save 3D model in requested format
        model_filename = f"model.{output_format}"
        model_path = os.path.join(output_dir, model_filename)
        
        if output_format.lower() == "obj":
            save_obj(mesh, model_path, colored=colored)
        elif output_format.lower() == "ply":
            save_ply(mesh, model_path, colored=colored)
        elif output_format.lower() == "gltf":
            save_gltf(mesh, model_path, colored=colored)
        else:
            raise HTTPException(status_code=400, detail="Unsupported output format")
        
        # Generate preview image
        preview_path = os.path.join(output_dir, "preview.png")
        generate_preview(mesh, preview_path)
        
        # Move files to permanent storage (you might want to use S3 here)
        permanent_dir = "/tmp/outputs"
        os.makedirs(permanent_dir, exist_ok=True)
        
        import uuid
        unique_id = str(uuid.uuid4())
        
        final_model_path = os.path.join(permanent_dir, f"{unique_id}_{model_filename}")
        final_preview_path = os.path.join(permanent_dir, f"{unique_id}_preview.png")
        
        shutil.move(model_path, final_model_path)
        shutil.move(preview_path, final_preview_path)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # Prepare response
        response_data = {
            "success": True,
            "message": "3D model generated successfully",
            "model_url": f"/download/{unique_id}_{model_filename}",
            "preview_url": f"/download/{unique_id}_preview.png",
            "metadata": {
                "format": output_format,
                "quality": quality,
                "colored": colored,
                "vertices": len(mesh.verts_list()[0]) if hasattr(mesh, 'verts_list') else 0,
                "faces": len(mesh.faces_list()[0]) if hasattr(mesh, 'faces_list') else 0
            }
        }
        
        return GenerationResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error generating 3D model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate_from_base64", response_model=GenerationResponse)
async def generate_from_base64(request: GenerationRequest):
    """Generate 3D model from base64 encoded image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name
        
        # Convert to UploadFile-like object
        with open(temp_path, "rb") as f:
            from fastapi import UploadFile
            upload_file = UploadFile(filename="image.png", file=f)
            
            result = await generate_3d_model(
                image=upload_file,
                quality=request.quality,
                colored=request.colored,
                output_format=request.output_format
            )
        
        # Clean up
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files"""
    file_path = os.path.join("/tmp/outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

def save_obj(mesh, path: str, colored: bool = True):
    """Save mesh as OBJ file"""
    vertices = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    
    with open(path, 'w') as f:
        # Write vertices
        for v in vertices:
            if colored and hasattr(mesh, 'textures') and mesh.textures is not None:
                # If texture information is available, you can add color here
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def save_ply(mesh, path: str, colored: bool = True):
    """Save mesh as PLY file"""
    vertices = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colored:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices
        for i, v in enumerate(vertices):
            if colored:
                # Default color or extract from texture if available
                color = [128, 128, 128]  # Default gray
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
            else:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def save_gltf(mesh, path: str, colored: bool = True):
    """Save mesh as GLTF file (basic implementation)"""
    # For a complete GLTF implementation, you might want to use pygltflib
    # This is a simplified version
    import json
    
    vertices = mesh.verts_list()[0].cpu().numpy()
    faces = mesh.faces_list()[0].cpu().numpy()
    
    # Create basic GLTF structure
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3"
            },
            {
                "bufferView": 1,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(faces) * 3,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(vertices) * 3 * 4
            },
            {
                "buffer": 0,
                "byteOffset": len(vertices) * 3 * 4,
                "byteLength": len(faces) * 3 * 2
            }
        ],
        "buffers": [{
            "byteLength": len(vertices) * 3 * 4 + len(faces) * 3 * 2
        }]
    }
    
    with open(path, 'w') as f:
        json.dump(gltf, f, indent=2)

def generate_preview(mesh, path: str):
    """Generate a preview image of the 3D mesh"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        vertices = mesh.verts_list()[0].cpu().numpy()
        faces = mesh.faces_list()[0].cpu().numpy()
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the mesh
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, alpha=0.8, cmap='viridis')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Model Preview')
        
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not generate preview: {e}")
        # Create a simple placeholder image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (400, 400), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((200, 200), "3D Model Generated", fill='black', anchor='mm')
        img.save(path)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )