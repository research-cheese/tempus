from PIL import Image
import numpy as np
from skimage.measure import label, regionprops

from utils.object_detection.bounding_box import BoundingBox
from utils.preprocessing.filter_colors import filter_colors
from utils.preprocessing.ground_truths import GroundTruths

CONNECTIVITY_4 = 1
CONNECTIVITY_8 = 2

def get_ground_truth(image_path, colors, group_disjoint = False, min_count = 10):
    """
    Returns 2D numpy array of 0s and 1s

    Args:
    image_path: str. Path to image.
    colors: list of tuples of RGB colors. These are the colors that will be considered as 1.
    """

    ground_truths = []

    for color in colors:
        ground_truth_pil = filter_colors(image_path, [color])

        # Get bounding boxes
        labeled = np.array(label(ground_truth_pil, return_num=True, connectivity=CONNECTIVITY_8)[0])

        if group_disjoint and labeled.max() > 5: continue
        if group_disjoint: labeled = np.where(labeled > 0, 1, 0)

        for i in range(1, labeled.max() + 1):
            # Convert all i to 1
            ground_truth = np.where(labeled == i, 1, 0)
            
            # Get bounding box
            regions = regionprops(ground_truth)
            y_min, x_min, y_max, x_max = regions[0].bbox

            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_size = bbox_width * bbox_height

            if ground_truth.sum() < min_count: continue
            if bbox_size < 5: continue
            if bbox_width < 2: continue
            if bbox_height < 2: continue

            ground_truths.append(
                GroundTruths(
                    bounding_box=BoundingBox(xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max),
                    segmentation=ground_truth,
                    base_image=Image.open(image_path)
                )
            )

    return ground_truths
