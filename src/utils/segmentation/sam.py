from typing import List

from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
import torch 

from utils.segmentation.segmentation_model import SegmentationModel
from utils.segmentation.segmentation_result import SegmentationResult
from utils.object_detection.bounding_box import BoundingBox

class SamSegmentationModel(SegmentationModel):
    def segment(self, image: Image, boxes: BoundingBox) -> List[SegmentationResult]:
        pass