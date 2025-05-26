import numpy as np
import cv2
from PIL import Image
import os
import platform
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ml_inpainting")

# Cache for pre-trained models
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".image_restoration_models")

def ensure_model_dir():
    """Ensure the model cache directory exists"""
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)
    return MODEL_CACHE_DIR

class MLInpainter:
    """Lightweight ML-based inpainting for historical image restoration"""
    
    def __init__(self, use_gpu=False):
        """Initialize the inpainter
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.model_loaded = False
        self.model_cache_dir = ensure_model_dir()
        self.has_onnx = self._check_onnx_available()
        
    def _check_gpu_available(self):
        """Check if CUDA-capable GPU is available"""
        try:
            cv_build_info = cv2.getBuildInformation()
            return "CUDA" in cv_build_info and "YES" in cv_build_info.split("CUDA")[1].split("\n")[0]
        except:
            return False
            
    def _check_onnx_available(self):
        """Check if ONNX runtime is available"""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False

    def _get_system_info(self):
        """Get basic system information to determine appropriate model size"""
        try:
            import psutil
            ram = psutil.virtual_memory().total / (1024**3)  # RAM in GB
        except ImportError:
            # Fallback if psutil is not available
            if platform.system() == 'Windows':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUS(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', c_ulong),
                        ('dwMemoryLoad', c_ulong),
                        ('dwTotalPhys', c_ulong),
                        ('dwAvailPhys', c_ulong),
                        ('dwTotalPageFile', c_ulong),
                        ('dwAvailPageFile', c_ulong),
                        ('dwTotalVirtual', c_ulong),
                        ('dwAvailVirtual', c_ulong)
                    ]
                memory_status = MEMORYSTATUS()
                memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
                kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
                ram = memory_status.dwTotalPhys / (1024**3)
            else:
                # Rough estimate for Unix-like systems
                ram = 4.0  # Default to 4GB
        
        return {
            'ram': ram,
            'platform': platform.system(),
            'processor': platform.processor()
        }

    def _load_appropriate_model(self):
        """Load the most appropriate model based on system capabilities"""
        system_info = self._get_system_info()
        
        # For systems with ONNX runtime and sufficient RAM, try stable diffusion lite
        if self.has_onnx and system_info['ram'] >= 6:
            logger.info("Attempting to load Stable Diffusion Lite inpainting model")
            try:
                self._load_sd_lite_model()
                return
            except Exception as e:
                logger.warning(f"Failed to load Stable Diffusion Lite model: {e}")
                # Fall back to simpler models
        
        # For low-end systems (less than 8GB RAM)
        if system_info['ram'] < 8:
            logger.info("Loading lightweight inpainting model (compatible with systems with < 8GB RAM)")
            self._load_lightweight_model()
        else:
            logger.info("Loading standard inpainting model")
            self._load_standard_model()
        
        self.model_loaded = True

    def _download_model_if_needed(self, model_filename, model_url):
        """Download pre-trained model if not already in cache"""
        model_path = os.path.join(self.model_cache_dir, model_filename)
        
        if not os.path.exists(model_path):
            logger.info(f"Downloading model from {model_url}...")
            try:
                # Try using urllib first
                import urllib.request
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("Model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download model with urllib: {e}")
                # Fallback to subprocess for systems without urllib
                try:
                    if platform.system() == 'Windows':
                        subprocess.run(f'powershell -Command "Invoke-WebRequest -Uri {model_url} -OutFile {model_path}"', shell=True)
                    else:
                        subprocess.run(f'curl -L {model_url} -o {model_path}', shell=True)
                    logger.info("Model downloaded successfully using subprocess")
                except Exception as e:
                    logger.error(f"All download methods failed: {e}")
                    raise RuntimeError("Failed to download model")
        
        return model_path

    def _load_sd_lite_model(self):
        """Load a lightweight version of Stable Diffusion for inpainting"""
        try:
            import onnxruntime as ort
            
            # Download the lightweight SD inpainting model
            model_path = self._download_model_if_needed(
                "sd_lite_inpainting.onnx", 
                "https://huggingface.co/carolineec/informative-drawings/resolve/main/model.onnx"
            )
            
            # Choose execution provider based on hardware
            providers = ['CPUExecutionProvider']
            if self.use_gpu:
                providers = ['CUDAExecutionProvider'] + providers
            
            # Create inference session with optimizations for low-memory systems
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = False  # Can help with memory usage
            session_options.enable_mem_reuse = True  # Reuse memory where possible
                
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            # Log model inputs and outputs
            model_inputs = self.session.get_inputs()
            model_outputs = self.session.get_outputs()
            logger.info(f"SD Lite model loaded: {len(model_inputs)} inputs, {len(model_outputs)} outputs")
            
            self.model_type = "sd_lite"
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading SD Lite model: {e}")
            raise

    def _load_lightweight_model(self):
        """Load a very lightweight model suitable for low-end computers"""
        # We'll use OpenCV's built-in models which are much lighter
        self.model_type = "opencv_telea"
        # No actual model loading needed - we'll use OpenCV's algorithm
        
    def _load_standard_model(self):
        """Load a standard model for better quality on capable systems"""
        # For standard systems, we still use OpenCV's inpainting but with better parameters
        self.model_type = "opencv_ns"

    def inpaint(self, image, mask, radius=5):
        if not self.model_loaded:
            self._load_appropriate_model()
        
        # Convert PIL images to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Ensure mask is properly formatted for inpainting
        if isinstance(mask, Image.Image):
            mask_cv = np.array(mask.convert('L'))
        else:
            mask_cv = mask
            
        # Threshold to ensure binary mask (in case it's not already)
        _, mask_cv = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)
        
        if self.model_type == "sd_lite":
            try:
                result = self._apply_sd_inpainting(image_cv, mask_cv)
            except Exception as e:
                logger.error(f"SD inpainting failed, falling back to OpenCV: {e}")
                # Fall back to OpenCV method
                result = cv2.inpaint(image_cv, mask_cv, radius, cv2.INPAINT_TELEA)
        elif self.model_type == "opencv_telea":
            # Fast Marching Method (Telea) - very lightweight
            result = cv2.inpaint(image_cv, mask_cv, radius, cv2.INPAINT_TELEA)
        elif self.model_type == "opencv_ns":
            # Navier-Stokes - better quality but still lightweight
            result = cv2.inpaint(image_cv, mask_cv, radius, cv2.INPAINT_NS)
        else:
            # Default to TELEA if model_type not recognized
            result = cv2.inpaint(image_cv, mask_cv, radius, cv2.INPAINT_TELEA)
        
        # Add color correction
        result = self._enhance_colors(result)
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    def _apply_sd_inpainting(self, image, mask):
        """Apply Stable Diffusion inpainting"""
        # Preprocess inputs
        h, w = image.shape[:2]
        
        # Resize to model input size
        input_size = (512, 512)  # Standard input size for SD models
        image_resized = cv2.resize(image, input_size)
        mask_resized = cv2.resize(mask, input_size)
        
        # Prepare inputs for the model
        # Normalize image to [-1, 1]
        image_norm = image_resized.astype(np.float32) / 127.5 - 1.0
        
        # Convert to NCHW format (batch, channels, height, width)
        image_nchw = np.transpose(image_norm, (2, 0, 1))[np.newaxis, ...]
        
        # Normalize mask to [0, 1]
        mask_norm = mask_resized.astype(np.float32) / 255.0
        mask_nchw = mask_norm[np.newaxis, np.newaxis, ...]
        
        # Create input dictionary - adapt based on your model's expected inputs
        input_dict = {
            'image': image_nchw,
            'mask': mask_nchw,
            'prompt': np.array(["restore old historical photo"], dtype=np.object_),
            'guidance_scale': np.array([7.5], dtype=np.float32),
            'steps': np.array([20], dtype=np.int32)
        }
        
        # Run inference
        outputs = self.session.run(None, input_dict)
        
        # Process output
        output_image = outputs[0][0]  # First output, first batch
        
        # Convert from [-1, 1] to [0, 255]
        output_image = ((output_image + 1) * 127.5).astype(np.uint8)
        
        # Convert from CHW to HWC
        output_image = np.transpose(output_image, (1, 2, 0))
        
        # Resize back to original dimensions
        result = cv2.resize(output_image, (w, h))
        
        return result
    
    def _enhance_colors(self, image_cv):
        """Enhance colors in the inpainted image"""
        # Apply color correction/harmonization
        lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into different channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L-channel with the A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr

def get_best_inpainter(use_gpu=False):
    """Factory function to get the best inpainter for the system"""
    system_info = {
        'ram': 4.0,  # Default assumption
        'has_gpu': False
    }
    
    # Try to get actual system info
    try:
        import psutil
        system_info['ram'] = psutil.virtual_memory().total / (1024**3)  # RAM in GB
    except ImportError:
        pass
    
    # Check for GPU capability
    try:
        cv_build_info = cv2.getBuildInformation()
        system_info['has_gpu'] = "CUDA" in cv_build_info and "YES" in cv_build_info.split("CUDA")[1].split("\n")[0]
    except:
        pass
    
    # Create and return appropriate inpainter
    return MLInpainter(use_gpu=use_gpu and system_info['has_gpu'])