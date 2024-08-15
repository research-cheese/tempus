from PIL import Image
import numpy as np

def filter_colors(image_path, colors, replace_id = 1):
    """
    Returns 2D numpy array of 0s and 1s
    """
    ground_truth_pil = Image.open(image_path).convert("RGB")
    ground_truth_pil = np.asarray(ground_truth_pil)

    # Iterate through ground truth pil and if color is in ground_truth_colors, set to 1, else 0.
    # It is a 2x2 image
    ground_truth_pil = np.array([[
        replace_id if tuple(pixel) in colors else 0
        for pixel in row
    ] for row in ground_truth_pil])
    
    return ground_truth_pil