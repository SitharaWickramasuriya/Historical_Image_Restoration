import customtkinter as ctk
from image_widgets import *
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
from menu import MenuPanel
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from quality_metrics import calculate_all_metrics
# Import our inpainting module
from ml_inpainting import get_best_inpainter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

class App(ctk.CTk):
    def __init__(self):
        # Setup
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.title('Historical Image Restoration in Sri Lanka')
        self.minsize(800, 500)
        self.init_parameters()

        # Center the window
        width, height = 1000, 600
        screen_width = self.winfo_screenwidth()  
        screen_height = self.winfo_screenheight()  
        x_coordinate = int((screen_width - width) / 2) 
        y_coordinate = int((screen_height - height) / 2)  
        self.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}") 

        # Layout
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=2, uniform='a')
        self.columnconfigure(1, weight=6, uniform='a')

        # Canvas data
        self.image_width = 0
        self.image_height = 0
        self.canvas_width = 0
        self.canvas_height = 0

        # Initialize inpainter (will select the appropriate model based on hardware)
        self.initialize_ml_model()

        # Widgets
        self.image_import = ImageImport(self, self.import_image)

    def initialize_ml_model(self):
        """Initialize the most appropriate ML model for inpainting based on the system capabilities"""
        try:
            # Use GPU if system supports it
            self.inpainter = get_best_inpainter(use_gpu=True)
            logger.info("ML inpainter initialized successfully")
            
            # Check if ONNX runtime is available for SD Lite
            self.has_sd_lite = False
            try:
                import onnxruntime
                self.has_sd_lite = True
                logger.info("ONNX Runtime available - Stable Diffusion Lite inpainting enabled")
            except ImportError:
                logger.info("ONNX Runtime not available")
                
        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")
            # Create a dummy inpainter in case of failure
            self.inpainter = None
            self.has_sd_lite = False

    def init_parameters(self):
        self.brush_settings = {
            'size': ctk.IntVar(value=PENSIZE_DEFAULT),
            'color': ctk.StringVar(value=PENCOLOR_DEFAULT)
        }
        self.hsv_vars = {
            'hue': ctk.DoubleVar(value=HUE_DEFAULT),
            'saturation': ctk.DoubleVar(value=SATURATION_DEFAULT),
            'value': ctk.DoubleVar(value=VALUE_DEFAULT)
        }
        for var in self.hsv_vars.values():
            var.trace('w', self.hsv_modified_image)
        
        # Add inpainting parameters
        self.inpaint_params = {
            'method': ctk.StringVar(value="ML"),  # ML, TELEA, or NS
            'radius': ctk.IntVar(value=5),        # Inpainting radius
        }

    def import_image(self, path):
        self.image_path = path
        self.original = Image.open(path)
        self.image = self.original.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.composite = self.image.copy()
        self.image_ratio = self.image.size[0] / self.image.size[1]
        self.image_tk = ImageTk.PhotoImage(self.image)

        self.image_import.grid_forget()

        self.image_output = ImageOutput(self, self.resize_image, self.brush_settings)
        self.image_output.set_original_image(self.original)
        self.image_output.grid(row=0, column=1, sticky="nsew")

        self.close_button = CloseOutput(self, self.close_edit)
        self.close_button.place(relx=0.99, rely=0.01, anchor="ne")

        # Update to pass inpainting parameters
        self.menu = MenuPanel(self, self.brush_settings, self.hsv_vars, self.image_output, self.inpaint_params)
        self.menu.grid(row=0, column=0, sticky="ns")

    def close_edit(self):
        self.image_output.grid_forget()
        self.close_button.place_forget()
        self.menu.grid_forget()
        self.image_import = ImageImport(self, self.import_image)

    def resize_image(self, event):
        canvas_ratio = event.width / event.height
        self.canvas_width = event.width
        self.canvas_height = event.height

        if canvas_ratio > self.image_ratio:
            self.image_height = int(event.height)
            self.image_width = int(self.image_height * self.image_ratio)
        else:
            self.image_width = int(event.width)
            self.image_height = int(self.image_width / self.image_ratio)

        self.place_image()

    def place_image(self):
        self.image_output.delete('all')
        resized_image = self.image.resize((self.image_width, self.image_height))
        self.image_tk = ImageTk.PhotoImage(resized_image)
        x_position = (self.canvas_width - self.image_width) // 2
        y_position = (self.canvas_height - self.image_height) // 2
        self.image_output.create_image(x_position, y_position, image=self.image_tk, anchor='nw')
        self.image_output.set_image_dimensions(x_position, y_position, self.image_width, self.image_height)

    def revert_image(self, original_image):
        self.image = original_image.copy()
        self.composite = original_image.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.place_image()

    def hsv_modified_image(self, *args):
        composite = self.composite.copy()
        np_image = np.array(composite.convert('RGB'))
        hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)

        hMin = int(self.hsv_vars['hue'].get())
        sMin = int(self.hsv_vars['saturation'].get())
        vMin = int(self.hsv_vars['value'].get())
        hMax = 179  
        sMax = 255
        vMax = 255

        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        mask = cv2.inRange(hsv_image, lower, upper)
        result = cv2.bitwise_and(np_image, np_image, mask=mask)

        self.image = Image.fromarray(result)
        self.place_image()

    def generate_mask(self):
        composite = self.composite.copy()
        np_image = np.array(composite.convert('RGB'))
        hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)

        hMin = int(self.hsv_vars['hue'].get())
        sMin = int(self.hsv_vars['saturation'].get())
        vMin = int(self.hsv_vars['value'].get())
        hMax = 179
        sMax = 255
        vMax = 255

        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        mask = cv2.inRange(hsv_image, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.mask = mask

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_pil = Image.fromarray(mask_rgb)
        self._show_mask_popup(mask_pil)
        return mask_pil

    def _show_mask_popup(self, mask_pil):
        popup = ctk.CTkToplevel(self)
        popup.title("Generated HSV Mask")
        popup.geometry("400x400")

        img_resized = mask_pil.resize((400, 400))
        mask_tk = ImageTk.PhotoImage(img_resized)

        label = ctk.CTkLabel(popup, image=mask_tk)
        label.image = mask_tk
        label.pack(pady=10)
        
        # Add an explanation label
        explanation = ctk.CTkLabel(popup, text="White areas will be inpainted/restored", 
                                 font=ctk.CTkFont(size=14))
        explanation.pack(pady=5)
        
        # Add close button
        close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)

    def apply_inpainting(self, method="ML", inpaint_radius=5):
        if not hasattr(self, "mask"):
            logger.warning("No mask available. Generate a mask first.")
            self._show_no_mask_warning()
            return

        original_pil = self.original.copy()
        mask_cv = self.mask.astype(np.uint8)
        
        try:
            # Status popup with appropriate message based on method
            if method.upper() == "ML" and self.has_sd_lite:
                self._show_processing_popup("Applying Stable Diffusion inpainting...\nThis may take a minute.")
            else:
                self._show_processing_popup("Applying inpainting...")
            
            if method.upper() == "ML" and self.inpainter is not None:
                # Use our ML-based inpainting
                inpainted_pil = self.inpainter.inpaint(original_pil, mask_cv, inpaint_radius)
            elif method.upper() == "" or (method.upper() == "ML" and self.inpainter is None):
                # Use OpenCV's TELEA method as fallback
                original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                inpainted_cv = cv2.inpaint(original_cv, mask_cv, inpaint_radius, cv2.INPAINT_TELEA)
                inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB))
            elif method.upper() in ["NS", "NAVIER-STOKES"]:
                # Use OpenCV's Navier-Stokes method
                original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                inpainted_cv = cv2.inpaint(original_cv, mask_cv, inpaint_radius, cv2.INPAINT_NS)
                inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB))
            else:
                # Default to TELEA if method is not recognized
                original_cv = cv2.cvtColor(np.array(original_pil), cv2.COLOR_RGB2BGR)
                inpainted_cv = cv2.inpaint(original_cv, mask_cv, inpaint_radius, cv2.INPAINT_TELEA)
                inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB))

            self.image = inpainted_pil.copy()
            self.composite = inpainted_pil.copy()
            self.draw = ImageDraw.Draw(self.image)
            self.place_image()

            # Close the processing popup
            if hasattr(self, 'processing_popup'):
                self.processing_popup.destroy()

            # Show what inpainting model was used
            inpainting_type = "Standard"
            if hasattr(self.inpainter, 'model_type'):
                if self.inpainter.model_type == "sd_lite":
                    inpainting_type = "Stable Diffusion Lite"
                elif self.inpainter.model_type == "opencv_ns":
                    inpainting_type = "OpenCV telea"
                elif self.inpainter.model_type == "opencv_telea":
                    inpainting_type = "Stable Diffusion Lite"
            
            # Show a success message with the model used
            self._show_success_popup(f"Inpainting successful using {inpainting_type} model")

            # Calculate metrics
            try:
                metrics = calculate_all_metrics(self.original, self.image)
                ssim_score = metrics.get("SSIM", "N/A")
                logger.info(f"SSIM Score: {ssim_score}")
                self.display_metrics(metrics)
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
                
        except Exception as e:
            logger.error(f"Error during inpainting: {e}")
            if hasattr(self, 'processing_popup'):
                self.processing_popup.destroy()
            self._show_error_popup(f"Inpainting failed: {str(e)}")

    def _show_processing_popup(self, message="Processing..."):
        """Show a popup indicating that processing is happening"""
        self.processing_popup = ctk.CTkToplevel(self)
        self.processing_popup.title("Processing")
        self.processing_popup.geometry("300x150")
        
        # Make it appear on top
        self.processing_popup.attributes('-topmost', True)
        
        label = ctk.CTkLabel(self.processing_popup, text=message, 
                            font=ctk.CTkFont(size=16))
        label.pack(pady=30)
        
        # Add a note about slow computers
        note = ctk.CTkLabel(
            self.processing_popup, 
            text="This may take longer on slower computers",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        note.pack(pady=5)
        
        # Update the UI to show the popup
        self.processing_popup.update()

    def _show_success_popup(self, message):
        """Show a success popup with the given message"""
        popup = ctk.CTkToplevel(self)
        popup.title("Success")
        popup.geometry("400x150")
        
        # Make it appear on top
        popup.attributes('-topmost', True)
        
        label = ctk.CTkLabel(popup, text=message, 
                            font=ctk.CTkFont(size=16))
        label.pack(pady=30)
        
        button = ctk.CTkButton(popup, text="OK", command=popup.destroy)
        button.pack(pady=10)

    def _show_error_popup(self, message):
        """Show an error popup with the given message"""
        popup = ctk.CTkToplevel(self)
        popup.title("Error")
        popup.geometry("400x150")
        
        label = ctk.CTkLabel(popup, text=message, font=ctk.CTkFont(size=16))
        label.pack(pady=30)
        
        button = ctk.CTkButton(popup, text="OK", command=popup.destroy)
        button.pack(pady=10)

    def _show_no_mask_warning(self):
        """Show a warning that no mask is available"""
        popup = ctk.CTkToplevel(self)
        popup.title("No Mask Available")
        popup.geometry("400x150")
        
        label = ctk.CTkLabel(popup, text="Please generate a mask first using HSV controls.", 
                            font=ctk.CTkFont(size=16))
        label.pack(pady=30)
        
        button = ctk.CTkButton(popup, text="OK", command=popup.destroy)
        button.pack(pady=10)

    def enhance_current_image(self):
        """Enhancement method using basic techniques"""
        if not hasattr(self, "image"):
            logger.warning("No image available to enhance.")
            return
        
        img = self.image.copy()

        # Apply a series of enhancements
        sharpener = ImageEnhance.Sharpness(img)
        img = sharpener.enhance(2.0)

        brightener = ImageEnhance.Brightness(img)
        img = brightener.enhance(1.2)

        contraster = ImageEnhance.Contrast(img)
        img = contraster.enhance(1.3)

        # Apply adaptive histogram equalization for better color balance
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        
        # Split the LAB channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the enhanced L channel with the original A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR and then to RGB
        img_cv = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL image
        img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        self.image = img.copy()
        self.composite = img.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.place_image()
        
        # Show success message
        self._show_success_popup("Image enhanced successfully")

    def display_metrics(self, metrics):
        """Display restoration quality metrics in a popup"""
        popup = ctk.CTkToplevel(self)
        popup.title("Image Quality Metrics")
        popup.geometry("400x150")

        # Format metrics for display
        formatted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics[key] = f"{value:.4f}"
            else:
                formatted_metrics[key] = str(value)
        
        # Create a nice looking metrics display
        frame = ctk.CTkFrame(popup)
        frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Title
        title = ctk.CTkLabel(frame, text="Restoration Quality Metrics", 
                           font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(pady=(10, 20))
        
        # Metrics grid
        for i, (metric, value) in enumerate(formatted_metrics.items()):
            metric_name = metric
            if metric == "SSIM":
                metric_name = "SSIM (Structural Similarity)"
                
            # Create a frame for each metric
            metric_frame = ctk.CTkFrame(frame)
            metric_frame.pack(pady=5, padx=10, fill="x")
            
            # Label with metric name
            metric_label = ctk.CTkLabel(metric_frame, text=metric_name, anchor="w")
            metric_label.pack(side="left", padx=10, pady=5)
            
            # Value
            value_label = ctk.CTkLabel(metric_frame, text=value, anchor="e")
            value_label.pack(side="right", padx=10, pady=5)
        
        # Close button
        close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy)
        close_btn.pack(pady=10)

    def display_ssim_score(self, score):
        """Legacy method for displaying just the SSIM score"""
        metrics = {"SSIM": score}
        self.display_metrics(metrics)

if __name__ == "__main__":
    app = App()
    app.mainloop()