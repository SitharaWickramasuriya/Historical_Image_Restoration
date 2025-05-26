from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

def calculate_ssim_score(original_image, restored_image):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Args:
        original_image: PIL Image object of the original image
        restored_image: PIL Image object of the restored image
        
    Returns:
        float: SSIM score (higher is better, 1.0 is perfect)
    """
    # Convert images to numpy arrays
    original_array = np.array(original_image)
    restored_array = np.array(restored_image)
    
    # Make sure images are the same size
    if original_array.shape != restored_array.shape:
        restored_array = cv2.resize(restored_array, 
                                   (original_array.shape[1], original_array.shape[0]))
    
    # Convert images to grayscale
    original_gray = rgb2gray(original_array)
    restored_gray = rgb2gray(restored_array)

    # Compute SSIM between the two images
    score, _ = ssim(original_gray, restored_gray, full=True)
    return score

def calculate_all_metrics(original_image, restored_image=None):
    """
    Calculate all image quality metrics.
    
    Args:
        original_image: PIL Image object of the original image
        restored_image: PIL Image object of the restored image (optional)
        
    Returns:
        dict: Dictionary with all metrics
    """
    # If restored_image is not provided, just calculate SSIM with itself (=1.0)
    if restored_image is None:
        restored_image = original_image
        
    # Make sure images are in RGB format
    original_rgb = original_image.convert("RGB")
    restored_rgb = restored_image.convert("RGB")
    
    # Calculate SSIM score
    ssim_score = calculate_ssim_score(original_rgb, restored_rgb)
    
    return {
        "SSIM": ssim_score
    }