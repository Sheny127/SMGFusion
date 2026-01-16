import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def restore_color(fused_gray_path, original_vis_path, save_path):
    """
    Combines the color from the original visible image with the fused grayscale image.
    Principle: RGB -> YCrCb, replace Y channel with fused image, YCrCb -> RGB.
    """
    # 1. Read images
    # Read the fused grayscale image (0 flag for grayscale mode)
    fused_img = cv2.imread(fused_gray_path, 0)
    # Read the original visible color image
    vis_img = cv2.imread(original_vis_path)

    if fused_img is None:
        print(f"[Error] Cannot read fused image: {fused_gray_path}")
        return
    if vis_img is None:
        print(f"[Error] Cannot read visible image: {original_vis_path}")
        return

    # 2. Dimension Alignment
    # The network output might have slight dimension differences due to padding or resizing.
    # We resize the fused image to match the original visible image's dimensions.
    h, w = vis_img.shape[:2]
    if fused_img.shape != (h, w):
        fused_img = cv2.resize(fused_img, (w, h), interpolation=cv2.INTER_CUBIC)

    # 3. Color Space Conversion (BGR -> YCrCb)
    # OpenCV reads in BGR order by default
    vis_ycrcb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YCrCb)

    # 4. Split Channels
    y_channel, cr_channel, cb_channel = cv2.split(vis_ycrcb)

    # 5. Replace Y Channel
    # Note: The fused image acts as the new luminance information.
    # Direct replacement is usually sufficient.
    new_y_channel = fused_img

    # 6. Merge Channels
    merged_ycrcb = cv2.merge([new_y_channel, cr_channel, cb_channel])

    # 7. Convert back to BGR Space
    final_color_img = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

    # 8. Save
    cv2.imwrite(save_path, final_color_img)

def batch_process(fused_dir, vis_dir, save_dir):
    """
    Batch process a directory of images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get all image filenames
    img_names = os.listdir(fused_dir)
    # Filter for image extensions
    img_names = [x for x in img_names if x.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg'))]

    print(f"Starting processing. Found {len(img_names)} fused images...")

    for img_name in tqdm(img_names):
        fused_path = os.path.join(fused_dir, img_name)
        
        # Assumption: The filenames are the same. 
        # If they differ (e.g., fused image has a prefix), modify the logic here.
        # Example: if fused is 'fused_001.png' and vis is '001.png':
        # vis_name = img_name.replace('fused_', '') 
        vis_name = img_name 
        
        vis_path = os.path.join(vis_dir, vis_name)
        save_path = os.path.join(save_dir, img_name)

        if not os.path.exists(vis_path):
            # Try to find a file with the same name but different extension (e.g., png vs jpg)
            prefix = os.path.splitext(vis_name)[0]
            found = False
            for ext in ['.jpg', '.png', '.bmp']:
                temp_path = os.path.join(vis_dir, prefix + ext)
                if os.path.exists(temp_path):
                    vis_path = temp_path
                    found = True
                    break
            if not found:
                print(f"[Warning] Corresponding visible image not found: {vis_name}, skipping.")
                continue

        restore_color(fused_path, vis_path, save_path)

    print(f"Processing complete! Color results saved in: {save_dir}")

if __name__ == "__main__":
    # --- Configuration Paths ---
    # 1. Directory containing the fused grayscale images (model output)
    fused_gray_folder = r"result" 
    
    # 2. Directory containing the original visible (color) images
    original_vis_folder = r"vi"
    
    # 3. Directory to save the final colored results
    output_color_folder = "results/RGB"

    # --- Run ---
    batch_process(fused_gray_folder, original_vis_folder, output_color_folder)