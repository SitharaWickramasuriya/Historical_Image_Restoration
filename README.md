# 🖼️ Historical Image Restoration GUI
A powerful Python-based GUI tool to restore historical images, such as wall paintings, using a combination of manual masking, traditional OpenCV inpainting, and AI-based Stable Diffusion. The tool offers HSV-based mask generation, brush painting, inpainting options, and image quality assessment via SSIM.

# ✨ Features
🎨 Manual Painting: Mark damaged regions using a brush tool.

🌈 HSV Masking: Automatically generate masks by adjusting Hue, Saturation, and Value sliders.

🔧 Restoration Methods:

  Stable Diffusion (ML) – High-quality AI-based inpainting.

  Telea (Fast) – Fast OpenCV inpainting for quick fixes.

  NS (Native / Navier-Stokes) – Quality-focused OpenCV inpainting.

🔍 Quality Assessment: Uses SSIM (Structural Similarity Index) to evaluate restoration results.

🛠️ Enhancement Tools:

  Denoising (FastNLMeans)

  Sharpness Boost

💾 Save Results: Export restored and enhanced images.

# 🖥️ How It Works
Load Image
Upload a damaged historical image (e.g., mural, wall painting).

Mark Damage
  Use the brush or HSV sliders to highlight areas for inpainting.

Choose Method
  Select one of the three inpainting methods:

ML (Stable Diffusion) – Uses machine learning for photorealistic restoration.

Fast (Telea) – Faster, edge-aware OpenCV algorithm.

Quality (NS) – Navier-Stokes based method for smoother fills.

Apply Inpainting
  Hit "Apply Inpainting" to restore the selected regions.

Enhance & Save
  Improve image quality via denoising and sharpness adjustments. Save final output.

Evaluate
  Use the "Check Image Quality" button to calculate SSIM score vs. original image.

# 📷 GUI Layout
Left Panel: Tool controls (HSV sliders, brush settings, method selection, enhancement).

Right Panel: Interactive canvas to view and edit the image.

# 🧠 Tech Stack
Python

CustomTkinter – for modern GUI

OpenCV – inpainting & HSV masking

Stable Diffusion – ML-based inpainting (optional setup)

scikit-image – SSIM evaluation

Pillow – image processing

