import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance, ImageTk
import cv2
import numpy as np
import datetime
from customtkinter import CTkImage
from skimage.metrics import structural_similarity as ssim


class MenuPanel(ctk.CTkFrame):
    def __init__(self, master, brush_settings, hsv_vars, image_output, inpaint_params=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.brush_settings = brush_settings
        self.hsv_vars = hsv_vars
        self.image_panel = image_output
        self.original_image = master.original if hasattr(master, 'original') else None
        self.inpaint_params = inpaint_params or {}

        self.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        # Make the panel scrollable for more controls
        self.create_scrollable_frame()
    
    def create_scrollable_frame(self):
        """Create a scrollable frame for all controls"""
        # Create a scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=200, height=500)
        self.scrollable_frame.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Add controls to the scrollable frame
        row = 0
        ctk.CTkButton(self.scrollable_frame, text="Check Image Quality", command=self.show_ssim_score).grid(row=row, column=0, pady=(0, 10), sticky="ew")
        row += 1

        row = self._create_hsv_controls(row)
        row = self._create_brush_controls(row)
        row = self._create_inpainting_controls(row)
        row = self._create_restore_buttons(row)
        row = self._create_enhancement_controls(row)

    def _create_hsv_controls(self, start_row):
        ctk.CTkLabel(self.scrollable_frame, text="HSV Masking", font=("Arial", 14, "bold")).grid(row=start_row, column=0, pady=(10, 5), sticky="w")
        row = start_row + 1
        row = self._add_slider("Hue", self.hsv_vars["hue"], 0, 179, row)
        row = self._add_slider("Saturation", self.hsv_vars["saturation"], 0, 255, row)
        row = self._add_slider("Value", self.hsv_vars["value"], 0, 255, row)

        ctk.CTkButton(self.scrollable_frame, text="Generate HSV Mask", command=self.master.generate_mask).grid(row=row, column=0, pady=(5, 10), sticky="ew")
        return row + 1

    def _create_brush_controls(self, start_row):
        ctk.CTkLabel(self.scrollable_frame, text="Brush Settings", font=("Arial", 14, "bold")).grid(row=start_row, column=0, pady=(20, 5), sticky="w")
        row = start_row + 1
        row = self._add_slider("Brush Size", self.brush_settings["size"], 1, 50, row)
        
        return row + 1

    def _create_inpainting_controls(self, start_row):
        if not self.inpaint_params:
            return start_row
            
        ctk.CTkLabel(self.scrollable_frame, text="Inpainting Settings", font=("Arial", 14, "bold")).grid(row=start_row, column=0, pady=(20, 5), sticky="w")
        row = start_row + 1
        
        # Method selection
        ctk.CTkLabel(self.scrollable_frame, text="Inpainting Method").grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        method_frame = ctk.CTkFrame(self.scrollable_frame)
        method_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        
        methods = [
            ("ML", "ML (Auto)"),
            ("TELEA", "Fast"),
            ("NS", "Quality")
        ]
        
        for i, (method_code, method_name) in enumerate(methods):
            btn = ctk.CTkRadioButton(
                method_frame,
                text=method_name,
                value=method_code,
                variable=self.inpaint_params["method"]
            )
            btn.grid(row=i, column=0, padx=5, pady=2, sticky="w")
        
        row += 1
        
        # Radius slider
        row = self._add_slider("Inpaint Radius", self.inpaint_params["radius"], 1, 20, row)
        
        # Inpaint button
        ctk.CTkButton(
            self.scrollable_frame, 
            text="Apply Inpainting", 
            command=self._restore_image
        ).grid(row=row, column=0, pady=5, sticky="ew")
        
        return row + 1

    def _create_restore_buttons(self, start_row):
        ctk.CTkLabel(self.scrollable_frame, text="Reset Options", font=("Arial", 14, "bold")).grid(row=start_row, column=0, pady=(20, 5), sticky="w")
        row = start_row + 1
        ctk.CTkButton(
            self.scrollable_frame, 
            text="Revert to Original", 
            command=self._revert_image,
            fg_color="#c75d55",  # Reddish color for warning
            hover_color="#a74b43"
        ).grid(row=row, column=0, pady=5, sticky="ew")
        return row + 1

    def _create_enhancement_controls(self, start_row):
        ctk.CTkLabel(self, text="Enhancement", font=("Arial", 14, "bold")).grid(row=start_row, column=0, pady=(20, 5), sticky="w")
        row = start_row + 1
        ctk.CTkButton(self, text="Enhance & Save", command=self._enhance_and_save).grid(row=row, column=0, pady=5, sticky="ew")
        return row + 1

    def _add_slider(self, text, variable, from_, to, row):
        ctk.CTkLabel(self.scrollable_frame, text=text).grid(row=row, column=0, sticky="w", padx=5)
        row += 1
        
        # Create a frame for the slider and value display
        slider_frame = ctk.CTkFrame(self.scrollable_frame)
        slider_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=2)
        slider_frame.grid_columnconfigure(0, weight=4)
        slider_frame.grid_columnconfigure(1, weight=1)
        
        # Create the slider
        slider = ctk.CTkSlider(
            slider_frame, 
            from_=from_, 
            to=to, 
            variable=variable, 
            orientation="horizontal"
        )
        slider.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        # Add a value label that updates with the slider
        value_label = ctk.CTkLabel(slider_frame, text=f"{variable.get():.1f}" if isinstance(from_, float) else str(int(variable.get())))
        value_label.grid(row=0, column=1, padx=5)
        
        # Update the label when the slider changes
        def update_label(*args):
            if isinstance(from_, float):
                value_label.configure(text=f"{variable.get():.1f}")
            else:
                value_label.configure(text=str(int(variable.get())))
                
        variable.trace_add("write", update_label)
        
        return row + 1

    def _revert_image(self):
        if hasattr(self.master, 'revert_image') and self.original_image:
            self.master.revert_image(self.original_image)

    def _restore_image(self):
        if hasattr(self.master, 'apply_inpainting'):
            method = self.inpaint_params.get("method", ctk.StringVar()).get()
            radius = self.inpaint_params.get("radius", ctk.IntVar()).get()
            self.master.apply_inpainting(method=method, inpaint_radius=radius)

    def format_score(self, score):
        return f"{score:.2f}" if score is not None and score != "N/A" else "N/A"

    def show_ssim_score(self):
        try:
            pil_image = Image.open(self.master.image_path)
            pil_image = pil_image.convert("RGB")
            score = self.calculate_ssim_score(pil_image)
            messagebox.showinfo("SSIM Score", f"The image SSIM score is: {self.format_score(score)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute SSIM score:\n{e}")

    def calculate_ssim_score(self, image):
        try:
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
            original_cv = np.array(self.master.original)
            original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2GRAY)
            score, _ = ssim(original_cv, image_cv, full=True)
            return score
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            return "N/A"
        
    def _enhance_and_save(self):
        try:
            pil_image = self.master.image.convert("RGB")
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
            denoised_pil = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

            enhancer = ImageEnhance.Sharpness(denoised_pil)
            enhanced_image = enhancer.enhance(2.5)

            self.master.image = enhanced_image
            self.master.place_image()

            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")])
            if file_path:
                enhanced_image.save(file_path)

            ssim_score = self.calculate_ssim_score(enhanced_image.convert("RGB"))
            messagebox.showinfo("Quality Scores", f"SSIM: {self.format_score(ssim_score)}")
            print(f"✅ Enhanced! SSIM Score: {self.format_score(ssim_score)}")

        except Exception as e:
            messagebox.showerror("Enhancement Error", str(e))

    def _save_quality_log(self, ssim_score):
        log_entry = f"Date: {datetime.datetime.now()}\n"
        log_entry += f"SSIM Score: {ssim_score:.2f}\n"
        log_entry += f"Image Path: {self.master.image_path}\n"
        log_entry += "=" * 50 + "\n"

        with open("quality_log.txt", "a") as f:
            f.write(log_entry)