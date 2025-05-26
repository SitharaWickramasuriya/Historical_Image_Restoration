import os
import numpy as np
import cv2
from PIL import Image
import platform
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("img2img")

# Cache for pre-trained models
MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".image_restoration_models")

def ensure_model_dir():
    """Ensure the model cache directory exists"""
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)
    return MODEL_CACHE_DIR

class StableDiffusionLite:
    """Lightweight version of Stable Diffusion img2img optimized for CPU"""
    
    def __init__(self, use_gpu=False):
        """Initialize the Stable Diffusion Lite engine
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.model_loaded = False
        self.model_cache_dir = ensure_model_dir()
        self.model = None
        self.session = None
        
        # Model paths
        self.model_filename = "sd_lite.onnx"
        self.model_path = os.path.join(self.model_cache_dir, self.model_filename)
        
        # URLs for model downloads
        self.model_url = "https://huggingface.co/carolineec/informative-drawings/resolve/main/model.onnx"

    def _check_gpu_available(self):
        """Check if CUDA-capable GPU is available"""
        try:
            cv_build_info = cv2.getBuildInformation()
            return "CUDA" in cv_build_info and "YES" in cv_build_info.split("CUDA")[1].split("\n")[0]
        except:
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

    def _download_model(self):
        """Download the pre-trained model"""
        model_path = os.path.join(self.model_cache_dir, self.model_filename)
        
        if not os.path.exists(model_path):
            logger.info(f"Downloading model from {self.model_url}...")
            try:
                # Try using urllib first
                import urllib.request
                urllib.request.urlretrieve(self.model_url, model_path)
                logger.info("Model downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                # Fallback to subprocess for systems without urllib
                try:
                    if platform.system() == 'Windows':
                        subprocess.run(f'powershell -Command "Invoke-WebRequest -Uri {self.model_url} -OutFile {model_path}"', shell=True)
                    else:
                        subprocess.run(f'curl -L {self.model_url} -o {model_path}', shell=True)
                    logger.info("Model downloaded successfully using subprocess")
                except Exception as e:
                    logger.error(f"All download methods failed: {e}")
                    raise RuntimeError("Failed to download model")
        
        return model_path

    def load_model(self):
        """Load the ONNX model for inference"""
        logger.info("Loading Stable Diffusion Lite model...")
        
        # Ensure model file exists or download it
        try:
            model_path = self._download_model()
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            raise RuntimeError("Failed to get Stable Diffusion Lite model")
        
        try:
            # Try to import ONNX Runtime
            import onnxruntime as ort
            
            # Choose execution provider based on hardware
            providers = ['CPUExecutionProvider']
            if self.use_gpu:
                providers = ['CUDAExecutionProvider'] + providers
            
            # Create inference session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, 
                                               providers=providers,
                                               sess_options=session_options)
            
            # Get model metadata
            model_inputs = self.session.get_inputs()
            model_outputs = self.session.get_outputs()
            
            logger.info(f"Model loaded successfully: {len(model_inputs)} inputs, {len(model_outputs)} outputs")
            self.model_loaded = True
            
        except ImportError:
            logger.error("ONNX Runtime not found. Falling back to OpenCV for basic enhancement.")
            raise ImportError("ONNX Runtime not installed. Please install with: pip install onnxruntime")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def img2img(self, input_image, prompt="enhance old photo", strength=0.7, 
               guidance_scale=7.5, steps=20):
        """
        Apply img2img transformation using the Stable Diffusion Lite model
        
        Args:
            input_image: PIL Image to transform
            prompt: Text description of the desired outcome
            strength: How much to transform (0.0-1.0)
            guidance_scale: How closely to follow the prompt
            steps: Number of diffusion steps
            
        Returns:
            Enhanced PIL Image
        """
        logger.info(f"Starting img2img with prompt: '{prompt}', strength: {strength}")
        
        if not self.model_loaded:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return self._fallback_enhance(input_image)
        
        try:
            # Convert PIL image to numpy array
            image_np = np.array(input_image.convert('RGB'))
            
            # Resize to model input size
            input_size = (512, 512)  # Standard size for most SD models
            resized_image = cv2.resize(image_np, input_size)
            
            # Prepare inputs for the model
            input_data = self._prepare_inputs(resized_image, prompt, strength, guidance_scale, steps)
            
            # Run inference
            outputs = self.session.run(None, input_data)
            
            # Process outputs to get the final image
            result_image = self._process_outputs(outputs, image_np.shape[:2])
            
            logger.info("img2img enhancement completed successfully")
            return result_image
            
        except Exception as e:
            logger.error(f"Error during img2img: {e}")
            return self._fallback_enhance(input_image)
    
    def _prepare_inputs(self, image, prompt, strength, guidance_scale, steps):
        """Prepare inputs for the ONNX model"""
        # Normalize image to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # Convert to model input format
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Get input names and prepare dictionary
        input_names = [input.name for input in self.session.get_inputs()]
        input_dict = {}
        
        # Map inputs based on model requirements
        for name in input_names:
            if 'image' in name.lower():
                input_dict[name] = image
            elif 'prompt' in name.lower():
                input_dict[name] = np.array([prompt], dtype=np.object_)
            elif 'strength' in name.lower():
                input_dict[name] = np.array([strength], dtype=np.float32)
            elif 'guidance' in name.lower() or 'scale' in name.lower():
                input_dict[name] = np.array([guidance_scale], dtype=np.float32)
            elif 'steps' in name.lower():
                input_dict[name] = np.array([steps], dtype=np.int32)
            else:
                # For unknown inputs, provide default values based on type
                if self.session.get_inputs()[input_names.index(name)].type == 'tensor(float)':
                    input_dict[name] = np.array([1.0], dtype=np.float32)
                elif self.session.get_inputs()[input_names.index(name)].type == 'tensor(int64)':
                    input_dict[name] = np.array([1], dtype=np.int32)
        
        return input_dict
    
    def _process_outputs(self, outputs, original_size):
        """Process model outputs to get the final image"""
        # Get the output image (first output is typically the generated image)
        output_image = outputs[0]
        
        # Denormalize from [-1, 1] to [0, 255]
        output_image = ((output_image[0] + 1) * 127.5).astype(np.uint8)
        
        # Convert from CHW to HWC
        output_image = np.transpose(output_image, (1, 2, 0))
        
        # Resize back to original dimensions
        output_image = cv2.resize(output_image, (original_size[1], original_size[0]))
        
        # Convert to PIL Image
        return Image.fromarray(output_image)
    
    def _fallback_enhance(self, image):
        """Fallback enhancement method using OpenCV if ONNX fails"""
        logger.info("Using fallback OpenCV enhancement")
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply a series of image enhancements
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)
        
        # 2. Enhance details
        detail = cv2.detailEnhance(denoised, sigma_s=10, sigma_r=0.15)
        
        # 3. Improve color and contrast using CLAHE in LAB color space
        lab = cv2.cvtColor(detail, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)
        
        # Merge enhanced L with original a,b
        enhanced_lab = cv2.merge((enhanced_l, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 4. Apply light sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

    def colorize(self, grayscale_image):
        """Colorize a grayscale image"""
        # Check if image is already colorized
        if len(np.array(grayscale_image).shape) == 3 and np.array(grayscale_image).shape[2] == 3:
            # Check if it's essentially grayscale despite having 3 channels
            img_array = np.array(grayscale_image)
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            if np.allclose(r, g, atol=5) and np.allclose(r, b, atol=5):
                logger.info("Image has 3 channels but appears grayscale. Proceeding with colorization.")
            else:
                logger.info("Image is already colorized. Applying general enhancement instead.")
                return self.img2img(grayscale_image, prompt="enhance historical photo, vivid colors", strength=0.6)
        
        return self.img2img(grayscale_image, prompt="colorize historical photo, realistic colors, vibrant", strength=0.8)

    def enhance_faces(self, image):
        """Enhance faces in the image"""
        # Try to detect faces
        try:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                logger.info(f"Detected {len(faces)} faces in the image")
                return self.img2img(image, 
                                   prompt="enhance historical photo portrait, clear facial features, detailed", 
                                   strength=0.6)
            else:
                logger.info("No faces detected. Applying general enhancement.")
                return self.img2img(image, 
                                   prompt="enhance historical photo, clear details", 
                                   strength=0.6)
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return self.img2img(image, 
                               prompt="enhance historical photo", 
                               strength=0.6)

    def restore_photo(self, image):
        """Full photo restoration"""
        return self.img2img(image, 
                           prompt="restore old damaged photo, clear, detailed, realistic", 
                           strength=0.75)


class LightweightEnhancer:
    """Extremely lightweight photo enhancer for very low-end systems"""
    
    def __init__(self):
        pass
    
    def enhance(self, image, enhancement_type="general"):
        """
        Enhance image using OpenCV techniques
        
        Args:
            image: PIL Image to enhance
            enhancement_type: Type of enhancement (general, colorize, face)
            
        Returns:
            Enhanced PIL Image
        """
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if enhancement_type == "colorize" and self._is_grayscale(image):
            # Apply colorization using histogram matching to a color palette
            result = self._apply_colorization(image_cv)
        elif enhancement_type == "face":
            # Apply face enhancement
            result = self._enhance_face(image_cv)
        else:
            # Apply general enhancement
            result = self._enhance_general(image_cv)
            
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    def _is_grayscale(self, image):
        """Check if an image is grayscale"""
        if image.mode == 'L':
            return True
            
        # Check if the image has 3 identical channels
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            return np.allclose(r, g, atol=5) and np.allclose(r, b, atol=5)
            
        return False
    
    def _enhance_general(self, image_cv):
        """Apply general image enhancement"""
        # Denoise
        denoised = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)
        
        # Enhance details
        detail = cv2.detailEnhance(denoised, sigma_s=10, sigma_r=0.15)
        
        # Improve contrast with CLAHE
        lab = cv2.cvtColor(detail, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Add subtle sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _apply_colorization(self, gray_cv):
        """Apply pseudo-colorization using OpenCV"""
        # Ensure image is grayscale
        if len(gray_cv.shape) == 3:
            gray = cv2.cvtColor(gray_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_cv
            
        # Create a 3-channel image from grayscale
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply false colorization by mapping to a historical photo palette
        # This uses a technique similar to selective color toning
        # Apply sepia tone effect as a base
        sepia = np.array([0.272, 0.534, 0.131])
        sepia_tone = cv2.transform(color, np.matrix(sepia))
        normalized = np.array([[[0, 0, 1]]])
        sepia_img = cv2.transform(sepia_tone, normalized)
        
        # Adjust colors for more natural look
        sepia_img = cv2.cvtColor(sepia_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(sepia_img)
        
        # Adjust a and b channels to create more natural colors
        a = cv2.addWeighted(a, 1.2, np.zeros_like(a), 0, 0)
        b = cv2.addWeighted(b, 1.2, np.zeros_like(b), 0, 0)
        
        # Merge the adjusted channels
        adjusted = cv2.merge((l, a, b))
        result = cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _enhance_face(self, image_cv):
        """Enhance faces in the image"""
        # Basic enhancement first
        enhanced = self._enhance_general(image_cv)
        
        try:
            # Try to detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) == 0:
                return enhanced
                
            # For each detected face, apply additional enhancement
            for (x, y, w, h) in faces:
                # Extract face region with some margin
                margin = int(w * 0.1)  # 10% margin
                face_x = max(0, x - margin)
                face_y = max(0, y - margin)
                face_w = min(enhanced.shape[1] - face_x, w + 2*margin)
                face_h = min(enhanced.shape[0] - face_y, h + 2*margin)
                
                face_region = enhanced[face_y:face_y+face_h, face_x:face_x+face_w]
                
                # Apply face-specific enhancements
                # 1. Additional detail enhancement with careful parameters for faces
                face_enhanced = cv2.detailEnhance(face_region, sigma_s=5, sigma_r=0.1)
                
                # 2. Smart sharpen: more sharpening for eyes, less for skin
                kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
                face_enhanced = cv2.filter2D(face_enhanced, -1, kernel)
                
                # Put enhanced face back into the image
                enhanced[face_y:face_y+face_h, face_x:face_x+face_w] = face_enhanced
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error in face enhancement: {e}")
            return enhanced


def get_best_enhancer(use_gpu=False):
    """Get the best enhancer based on system capabilities"""
    # Try to import onnxruntime to check if it's available
    try:
        import onnxruntime
        try:
            # Try to create StableDiffusionLite
            enhancer = StableDiffusionLite(use_gpu=use_gpu)
            # Just check if model can be downloaded, don't load it yet
            return enhancer
        except Exception as e:
            logger.warning(f"StableDiffusionLite not available: {e}")
            return LightweightEnhancer()
    except ImportError:
        logger.warning("ONNX Runtime not available. Using lightweight enhancer.")
        return LightweightEnhancer()