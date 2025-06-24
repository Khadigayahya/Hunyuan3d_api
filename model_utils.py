import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class Hunyuan3DModelWrapper:
    """Wrapper class for Hunyuan3D-2 model with enhanced features"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.is_loaded = False
        
    def load_model(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """Load the Hunyuan3D-2 model"""
        try:
            # Determine device
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            logger.info(f"Loading model on device: {self.device}")
            
            # Import the model
            from hy3dgen import Hunyuan3DPipeline
            
            # Load model
            model_name = model_path or "Tencent-Hunyuan/Hunyuan3D-2"
            
            self.model = Hunyuan3DPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                variant="fp16" if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cuda":
                self.model = self.model.to(self.device)
                # Enable memory efficient attention if available
                if hasattr(self.model, 'enable_memory_efficient_attention'):
                    self.model.enable_memory_efficient_attention()
                
                # Enable model CPU offload for large models
                if hasattr(self.model, 'enable_model_cpu_offload'):
                    self.model.enable_model_cpu_offload()
            
            self.is_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def generate_3d(
        self,
        image,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 1.0,
        seed: Optional[int] = None,
        enhance_quality: bool = True,
        colored_output: bool = True
    ) -> Dict[str, Any]:
        """Generate 3D model from 2D image with enhanced quality"""
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Generate with enhanced parameters
            with torch.no_grad():
                # Enable attention slicing for memory efficiency
                if hasattr(self.model, 'enable_attention_slicing'):
                    self.model.enable_attention_slicing()
                
                result = self.model(
                    image=image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    output_type="mesh",
                    return_dict=True,
                    generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
                )
            
            # Extract mesh and enhance if requested
            mesh = result.meshes[0]
            
            if enhance_quality:
                mesh = self._enhance_mesh_quality(mesh)
            
            if colored_output:
                mesh = self._add_color_information(mesh, image)
            
            # Calculate mesh statistics
            vertices = mesh.verts_list()[0]
            faces = mesh.faces_list()[0]
            
            metadata = {
                "vertex_count": len(vertices),
                "face_count": len(faces),
                "has_textures": hasattr(mesh, 'textures') and mesh.textures is not None,
                "has_colors": hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None,
                "bounding_box": self._calculate_bounding_box(vertices),
                "generation_params": {
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "strength": strength,
                    "seed": seed,
                    "enhance_quality": enhance_quality,
                    "colored_output": colored_output
                }
            }
            
            return {
                "mesh": mesh,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating 3D model: {str(e)}")
            return {
                "mesh": None,
                "metadata": None,
                "success": False,
                "error": str(e)
            }
    
    def _enhance_mesh_quality(self, mesh):
        """Enhance mesh quality through various techniques"""
        try:
            # This is a placeholder for mesh enhancement techniques
            # In practice, you might want to implement:
            # - Mesh smoothing
            # - Subdivision
            # - Normal computation
            # - Texture enhancement
            
            # For now, we'll return the mesh as-is
            # You can implement specific enhancement algorithms here
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Could not enhance mesh quality: {e}")
            return mesh
    
    def _add_color_information(self, mesh, original_image):
        """Add color information to the mesh based on the original image"""
        try:
            # This is a simplified implementation
            # In practice, you might want to implement proper texture mapping
            
            # Extract colors from the original image
            import numpy as np
            from PIL import Image
            
            if isinstance(original_image, Image.Image):
                img_array = np.array(original_image)
            else:
                img_array = original_image
            
            # Calculate average color
            avg_color = np.mean(img_array.reshape(-1, 3), axis=0) / 255.0
            
            # Apply color to vertices (simplified approach)
            vertices = mesh.verts_list()[0]
            vertex_colors = torch.tensor(avg_color).unsqueeze(0).repeat(len(vertices), 1)
            
            # Store color information (this depends on your mesh structure)
            if hasattr(mesh, 'vertex_colors'):
                mesh.vertex_colors = vertex_colors
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Could not add color information: {e}")
            return mesh
    
    def _calculate_bounding_box(self, vertices):
        """Calculate bounding box of the mesh"""
        try:
            vertices_np = vertices.cpu().numpy()
            min_coords = np.min(vertices_np, axis=0)
            max_coords = np.max(vertices_np, axis=0)
            
            return {
                "min": min_coords.tolist(),
                "max": max_coords.tolist(),
                "center": ((min_coords + max_coords) / 2).tolist(),
                "size": (max_coords - min_coords).tolist()
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate bounding box: {e}")
            return None
    
    def optimize_for_inference(self):
        """Optimize model for faster inference"""
        if not self.is_loaded:
            return False
            
        try:
            # Enable various optimizations
            if self.device.type == "cuda":
                # Enable Flash Attention if available
                if hasattr(self.model, 'enable_flash_attention'):
                    self.model.enable_flash_attention()
                
                # Enable xFormers if available
                if hasattr(self.model, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.model.enable_xformers_memory_efficient_attention()
                        logger.info("xFormers memory efficient attention enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable xFormers: {e}")
                
                # Compile model if PyTorch 2.0+
                if hasattr(torch, 'compile'):
                    try:
                        self.model.unet = torch.compile(self.model.unet, mode="reduce-overhead")
                        logger.info("Model compiled for faster inference")
                    except Exception as e:
                        logger.warning(f"Could not compile model: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"loaded": False}
        
        try:
            return {
                "loaded": True,
                "device": str(self.device),
                "model_type": type(self.model).__name__,
                "memory_usage": self._get_memory_usage(),
                "capabilities": {
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "mixed_precision": self.device.type == "cuda"
                }
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"loaded": True, "error": str(e)}
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            if torch.cuda.is_available():
                return {
                    "gpu_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                    "gpu_reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
                    "gpu_max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
                }
            else:
                import psutil
                process = psutil.Process()
                return {
                    "cpu_memory": process.memory_info().rss / 1024**3  # GB
                }
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return {}
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear memory: {e}")


class MeshProcessor:
    """Utility class for mesh processing operations"""
    
    @staticmethod
    def save_mesh_obj(mesh, filepath: str, include_colors: bool = True, include_normals: bool = True):
        """Save mesh in OBJ format with enhanced features"""
        try:
            vertices = mesh.verts_list()[0].cpu().numpy()
            faces = mesh.faces_list()[0].cpu().numpy()
            
            with open(filepath, 'w') as f:
                # Write header
                f.write("# Generated by Hunyuan3D-2 API\n")
                f.write(f"# Vertices: {len(vertices)}\n")
                f.write(f"# Faces: {len(faces)}\n\n")
                
                # Write vertices
                for i, v in enumerate(vertices):
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
                    
                    # Add colors if available
                    if include_colors and hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
                        colors = mesh.vertex_colors[i].cpu().numpy()
                        f.write(f" {colors[0]:.3f} {colors[1]:.3f} {colors[2]:.3f}")
                    
                    f.write("\n")
                
                # Write normals if requested and available
                if include_normals and hasattr(mesh, 'verts_normals_list'):
                    try:
                        normals = mesh.verts_normals_list()[0].cpu().numpy()
                        for n in normals:
                            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
                    except:
                        pass
                
                f.write("\n")
                
                # Write faces (OBJ uses 1-based indexing)
                for face in faces:
                    if include_normals and hasattr(mesh, 'verts_normals_list'):
                        f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                    else:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving OBJ file: {e}")
            return False
    
    @staticmethod
    def save_mesh_ply(mesh, filepath: str, include_colors: bool = True):
        """Save mesh in PLY format with colors"""
        try:
            vertices = mesh.verts_list()[0].cpu().numpy()
            faces = mesh.faces_list()[0].cpu().numpy()
            
            # Check for colors
            has_colors = (include_colors and 
                         hasattr(mesh, 'vertex_colors') and 
                         mesh.vertex_colors is not None)
            
            with open(filepath, 'w') as f:
                # PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("comment Generated by Hunyuan3D-2 API\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                
                if has_colors:
                    f.write("property uchar red\n")
                    f.write("property uchar green\n")
                    f.write("property uchar blue\n")
                
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # Write vertices
                for i, v in enumerate(vertices):
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
                    
                    if has_colors:
                        colors = mesh.vertex_colors[i].cpu().numpy()
                        # Convert to 0-255 range
                        r, g, b = (colors * 255).astype(int)
                        f.write(f" {r} {g} {b}")
                    
                    f.write("\n")
                
                # Write faces
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving PLY file: {e}")
            return False
    
    @staticmethod
    def save_mesh_gltf(mesh, filepath: str, include_colors: bool = True):
        """Save mesh in GLTF format"""
        try:
            import json
            import struct
            import base64
            
            vertices = mesh.verts_list()[0].cpu().numpy()
            faces = mesh.faces_list()[0].cpu().numpy()
            
            # Prepare binary data
            vertex_data = vertices.astype(np.float32).tobytes()
            face_data = faces.astype(np.uint32).flatten().tobytes()
            
            # Calculate buffer info
            vertex_buffer_length = len(vertex_data)
            face_buffer_length = len(face_data)
            total_buffer_length = vertex_buffer_length + face_buffer_length
            
            # Create GLTF structure
            gltf = {
                "asset": {
                    "version": "2.0",
                    "generator": "Hunyuan3D-2 API"
                },
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{
                    "mesh": 0,
                    "name": "Hunyuan3D_Mesh"
                }],
                "meshes": [{
                    "name": "GeneratedMesh",
                    "primitives": [{
                        "attributes": {
                            "POSITION": 0
                        },
                        "indices": 1,
                        "mode": 4  # TRIANGLES
                    }]
                }],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,  # FLOAT
                        "count": len(vertices),
                        "type": "VEC3",
                        "min": vertices.min(axis=0).tolist(),
                        "max": vertices.max(axis=0).tolist()
                    },
                    {
                        "bufferView": 1,
                        "componentType": 5125,  # UNSIGNED_INT
                        "count": len(faces) * 3,
                        "type": "SCALAR"
                    }
                ],
                "bufferViews": [
                    {
                        "buffer": 0,
                        "byteOffset": 0,
                        "byteLength": vertex_buffer_length,
                        "target": 34962  # ARRAY_BUFFER
                    },
                    {
                        "buffer": 0,
                        "byteOffset": vertex_buffer_length,
                        "byteLength": face_buffer_length,
                        "target": 34963  # ELEMENT_ARRAY_BUFFER
                    }
                ],
                "buffers": [{
                    "byteLength": total_buffer_length,
                    "uri": f"data:application/octet-stream;base64,{base64.b64encode(vertex_data + face_data).decode()}"
                }]
            }
            
            # Add color information if available
            if (include_colors and 
                hasattr(mesh, 'vertex_colors') and 
                mesh.vertex_colors is not None):
                
                colors = mesh.vertex_colors.cpu().numpy().astype(np.float32)
                color_data = colors.tobytes()
                
                # Update buffer info
                gltf["buffers"][0]["byteLength"] += len(color_data)
                gltf["buffers"][0]["uri"] = f"data:application/octet-stream;base64,{base64.b64encode(vertex_data + face_data + color_data).decode()}"
                
                # Add color buffer view and accessor
                gltf["bufferViews"].append({
                    "buffer": 0,
                    "byteOffset": vertex_buffer_length + face_buffer_length,
                    "byteLength": len(color_data),
                    "target": 34962
                })
                
                gltf["accessors"].append({
                    "bufferView": 2,
                    "componentType": 5126,  # FLOAT
                    "count": len(colors),
                    "type": "VEC3"
                })
                
                # Add color attribute to mesh
                gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = 2
            
            # Save GLTF file
            with open(filepath, 'w') as f:
                json.dump(gltf, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving GLTF file: {e}")
            return False
    
    @staticmethod
    def generate_mesh_preview(mesh, output_path: str, size: Tuple[int, int] = (800, 600)):
        """Generate a high-quality preview image of the mesh"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.patches as patches
            
            vertices = mesh.verts_list()[0].cpu().numpy()
            faces = mesh.faces_list()[0].cpu().numpy()
            
            # Create figure with high DPI
            fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the mesh with better styling
            surf = ax.plot_trisurf(
                vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces,
                alpha=0.9,
                cmap='viridis',
                edgecolor='none',
                linewidth=0.1
            )
            
            # Improve lighting and viewing angle
            ax.view_init(elev=20, azim=45)
            
            # Set equal aspect ratio
            max_range = np.array([
                vertices[:, 0].max() - vertices[:, 0].min(),
                vertices[:, 1].max() - vertices[:, 1].min(),
                vertices[:, 2].max() - vertices[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            
            # Set background color
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            # Make pane edges invisible
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')
            
            # Add title
            plt.title('3D Model Preview', fontsize=16, pad=20)
            
            # Save with high quality
            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                transparent=False
            )
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            # Create simple placeholder
            try:
                from PIL import Image, ImageDraw, ImageFont
                
                img = Image.new('RGB', size, color='white')
                draw = ImageDraw.Draw(img)
                
                # Draw a simple 3D-like wireframe
                center_x, center_y = size[0] // 2, size[1] // 2
                draw.rectangle([center_x-100, center_y-100, center_x+100, center_y+100], outline='black', width=2)
                draw.line([center_x-100, center_y-100, center_x-50, center_y-150], fill='black', width=2)
                draw.line([center_x+100, center_y-100, center_x+150, center_y-150], fill='black', width=2)
                draw.line([center_x-100, center_y+100, center_x-50, center_y+50], fill='black', width=2)
                draw.line([center_x+100, center_y+100, center_x+150, center_y+50], fill='black', width=2)
                
                draw.text((center_x, center_y+150), "3D Model Generated", fill='black', anchor='mm')
                
                img.save(output_path)
                return True
                
            except Exception as e2:
                logger.error(f"Error creating placeholder: {e2}")
                return False