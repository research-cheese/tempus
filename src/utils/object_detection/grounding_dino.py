from typing import List

import torch
from transformers import pipeline
from PIL import Image

from utils.object_detection.object_detection_model import GroundedObjectDetectionModel
from utils.object_detection.detection_result import DetectionResult
from utils.core.env_var import Environment


class GroundingDinoModel(GroundedObjectDetectionModel):
    def __init__(self, detector_id=None, threshold=0.5):
        self.detector_id = (
            detector_id
            if detector_id is not None
            else "IDEA-Research/grounding-dino-tiny"
        )
        self.threshold = threshold

        if Environment().device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, image: Image, text: str) -> List[DetectionResult]:
        object_detector = pipeline(
            model=self.detector_id, task="zero-shot-object-detection", device=self.device
        )

        results = object_detector(
            image, candidate_labels=[text], threshold=self.threshold
        )
        results = [DetectionResult.from_dict(result) for result in results]
        return results
