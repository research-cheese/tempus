from typing import List

from PIL import Image

from utils.segmentation.segmentation_model import SegmentationModel
from utils.segmentation.segmentation_result import SegmentationResult
from utils.object_detection.bounding_box import BoundingBox

class LoraSamSegmentationModel(SegmentationModel):
    def segment(self, image: Image, boxes: BoundingBox) -> List[SegmentationResult]:
        pass    