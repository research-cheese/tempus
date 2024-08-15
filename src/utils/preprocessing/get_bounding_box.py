from PIL import Image
import numpy as np
from skimage.measure import label, regionprops

from utils.preprocessing.filter_colors import filter_colors

CONNECTIVITY_4 = 1
CONNECTIVITY_8 = 2

def get_bounding_boxes(image_path, colors, group_pixels = False):
    """
    Returns 2D numpy array of 0s and 1s

    Args:
    image_path: str. Path to image.
    colors: list of tuples of RGB colors. These are the colors that will be considered as 1.
    """
    ground_truth_pil = filter_colors(image_path, colors)

    # Get bounding boxes
    labeled = np.array(label(ground_truth_pil, return_num=True, connectivity=CONNECTIVITY_4)[0])
    regions = regionprops(labeled)

    from matplotlib import pyplot as plt    
    plt.imshow(labeled, cmap='gray')
    plt.show()
    
    plt.imshow(labeled, cmap='gray')
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        plt.plot([minc, minc, maxc, maxc, minc], [minr, maxr, maxr, minr, minr], 'r', linewidth=2)
    plt.show()
