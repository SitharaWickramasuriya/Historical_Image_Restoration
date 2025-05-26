from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from tkinter import filedialog

def calculate_ssim_score(original_image, restored_image):
    # Convert images to grayscale
    original_gray = rgb2gray(np.array(original_image))
    restored_gray = rgb2gray(np.array(restored_image))

    # Compute SSIM between the two images
    score, _ = ssim(original_gray, restored_gray, full=True)
    return score

def on_check_quality_button_click(self):
    # Open the original and restored images from file paths
    original_image_path = filedialog.askopenfilename()  # Original image path
    restored_image_path = filedialog.askopenfilename()  # Restored image path

    original_image = Image.open(original_image_path)
    restored_image = Image.open(restored_image_path)

    # Now call the SSIM score function with both images
    score = calculate_ssim_score(original_image, restored_image)
    
    print(f"SSIM Score: {score}")
