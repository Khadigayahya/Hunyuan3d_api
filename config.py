import os
from typing import Dict, Any

class Config:
    """Configuration settings for the Hunyuan3D API"""

    # ===========================
    # Model settings
    # ===========================
    MODEL_NAME = "Tencent-Hunyuan/Hunyuan3D-2"
    MODEL_CACHE_DIR = "/app/models"

    # ===========================
    # Server settings
    # ===========================
    HOST = "0.0.0.0"
    PORT = 8080
    WORKERS = 1

    # ===========================
    # File handling
    # ===========================
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    OUTPUT_DIR = "/tmp/outputs"
    TEMP_DIR = "/tmp/hunyuan3d"

    # ===========================
    # Model generation settings
    # ===========================
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

    # ===========================
    # Performance settings
    # ===========================
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True
    ENABLE_CPU_OFFLOAD = True
    ENABLE_ATTENTION_SLICING = True
    ENABLE_XFORMERS = True

    # ===========================
    # Output format
    # ===========================
    SUPPORTED_FORMATS = ['obj', 'ply', 'glb']  # Note: 'glb' is binary glTF
    DEFAULT_FORMAT = 'obj'

    # ===========================
    # Preview image
    # ===========================
    PREVIEW_SIZE = (800, 600)
    PREVIEW_DPI = 150

    # ===========================
    # Logging
    # ===========================
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ===========================
    # CORS
    # ===========================
    CORS_ORIGINS = ["*"]
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]

    # ===========================
    # Health checks
    # ===========================
    HEALTH_CHECK_INTERVAL = 30
    HEALTH_CHECK_TIMEOUT = 30

    # ===========================
    # Auto-cleanup
    # ===========================
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
        """Ensure needed directories exist"""
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DIR, exist_ok=True)

    @classmethod
    def validate_file_extension(cls, filename: str) -> bool:
        """Check if the file extension is allowed"""
        ext = os.path.splitext(filename.lower())[1]
        return ext in cls.ALLOWED_EXTENSIONS

    @classmethod
    def get_output_path(cls, filename: str) -> str:
        """Full output file path"""
        return os.path.join(cls.OUTPUT_DIR, filename)

    @classmethod
    def get_temp_path(cls, filename: str = None) -> str:
        """Full temporary file path"""
        if filename:
            return os.path.join(cls.TEMP_DIR, filename)
        return cls.TEMP_DIR

# ============================================================
# Different configurations for various environments
# ============================================================

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"
    WORKERS = min(4, (os.cpu_count() or 1))

class SageMakerConfig(Config):
    """Configuration specific to AWS SageMaker deployment"""
    DEBUG = False
    LOG_LEVEL = "INFO"

    # SageMaker specific directories
    MODEL_DIR = "/opt/ml/model"
    INPUT_DIR = "/opt/ml/input"
    OUTPUT_DIR = "/opt/ml/output"

    @classmethod
    def setup_sagemaker_paths(cls):
        """Setup paths for SageMaker runtime"""
        cls.MODEL_CACHE_DIR = cls.MODEL_DIR
        cls.OUTPUT_DIR = "/tmp/outputs"  # Using /tmp for generated files
        cls.TEMP_DIR = "/tmp/hunyuan3d"
        cls.create_directories()

# ============================================================
# Helper to load configuration based on ENVIRONMENT variable
# ============================================================

def get_config():
    env = os.getenv('ENVIRONMENT', 'development').lower()
    if env == 'production':
        return ProductionConfig
    elif env == 'sagemaker':
        config = SageMakerConfig
        config.setup_sagemaker_paths()
        return config
    else:
        return DevelopmentConfig
