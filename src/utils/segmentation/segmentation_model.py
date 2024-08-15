from abc import ABC, abstractmethod
from typing import List
from PIL import Image

from utils.segmentation.segmentation_result import SegmentationResult
from utils.object_detection.bounding_box import BoundingBox

class SegmentationModel(ABC):

    @abstractmethod
    def segment(self, image_path: str, box: BoundingBox) -> List[SegmentationResult]:
        """
        Returns a list of objects detected in the image
        """
        pass

    @abstractmethod
    def cached_segment(self, image_path: str, box: BoundingBox) -> List[SegmentationResult]:
        """
        Returns a list of objects detected in the image
        """
        pass