import os
from typing import Dict, Any

class Config:
    """Configuration settings for the Hunyuan3D API"""
    
    # Model settings
    MODEL_NAME = "Tencent-Hunyuan/Hunyuan3D-2"
    MODEL_CACHE_DIR = "/app/models"
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8080
    WORKERS = 1
    
    # File settings
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    OUTPUT_DIR = "/tmp/outputs"
    TEMP_DIR = "/tmp/hunyuan3d"
    
    # Model generation settings
    DEFAULT_QUALITY_SETTINGS = {
        "high": {
            "resolution": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "strength": 1.0
        },
        "medium": {
            "resolution": 512,
            "num_inference_steps": 30,
            "guidance_scale": 7.0,
            "strength": 0.9
        },
        "low": {
            "resolution": 256,
            "num_inference_steps": 20,
            "guidance_scale": 6.5,
            "strength": 0.8
        }
    }
    
    # Memory optimization settings
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True
    ENABLE_CPU_OFFLOAD = True
    ENABLE_ATTENTION_SLICING = True
    ENABLE_XFORMERS = True
    
    # Output format settings
    SUPPORTED_FORMATS = ['obj', 'ply', 'gltf']
    DEFAULT_FORMAT = 'obj'
    
    # Preview settings
    PREVIEW_SIZE = (800, 600)
    PREVIEW_DPI = 150
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS settings
    CORS_ORIGINS = ["*"]
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Health check settings
    HEALTH_CHECK_INTERVAL = 30
    HEALTH_CHECK_TIMEOUT = 30
    
    # File cleanup settings
    CLEANUP_INTERVAL_HOURS = 24
    MAX_FILE_AGE_HOURS = 48
    
    @classmethod
    def get_quality_settings(cls, quality: str) -> Dict[str, Any]:
        """Get quality settings for the specified quality level"""
        return cls.DEFAULT_QUALITY_SETTINGS.get(
            quality.lower(), 
            cls.DEFAULT_QUALITY_SETTINGS["high"]
        )
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed"""
        ext = os.path.splitext(filename.lower())[1]
        return ext in cls.ALLOWED_EXTENSIONS
    
    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """Get full output path for a file"""
        return os.path.join(cls.OUTPUT_DIR, filename)
    
    @classmethod
    def get_temp_path(cls, filename: str = None) -> str:
        """Get temporary file path"""
        if filename:
            return os.path.join(cls.TEMP_DIR, filename)
        return cls.TEMP_DIR

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    WORKERS = min(4, (os.cpu_count() or 1))

class SageMakerConfig(Config):
    """SageMaker-specific configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"
    
    # SageMaker specific paths
    MODEL_DIR = "/opt/ml/model"
    INPUT_DIR = "/opt/ml/input"
    OUTPUT_DIR = "/opt/ml/output"
    
    # Override paths for SageMaker
    @classmethod
    def setup_sagemaker_paths(cls):
        """Setup paths for SageMaker deployment"""
        cls.MODEL_CACHE_DIR = cls.MODEL_DIR
        cls.OUTPUT_DIR = "/tmp/outputs"  # Use /tmp for temporary outputs
        cls.TEMP_DIR = "/tmp/hunyuan3d"
        cls.create_directories()

# Get configuration based on environment
def get_config():
    """Get configuration based on environment variable"""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return ProductionConfig
    elif env == 'sagemaker':
        config = SageMakerConfig
        config.setup_sagemaker_paths()
        return config
    else:
        return DevelopmentConfig