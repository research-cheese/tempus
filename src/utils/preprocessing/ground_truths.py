from utils.object_detection.bounding_box import BoundingBox

import numpy as np
from PIL import Image

class GroundTruths:
    name: str
    bounding_box: BoundingBox
    segmentation: np.array
    base_image: Image

    def __init__(self, bounding_box, segmentation, base_image):
        self.bounding_box = bounding_box
        self.segmentation = segmentation
        self.base_image = base_image

    def __str__(self):
        return f'GroundTruths(bounding_box={self.bounding_box}, segmentation={self.segmentation}, base_image={self.base_image})'