from abc import ABC, abstractmethod
from typing import List
from PIL import Image

from utils.object_detection.detection_result import DetectionResult


# class GroundedObjectDetectionModel(ABC):

#     @abstractmethod
#     def detect(self, image: Image, text: str) -> List[DetectionResult]:
#         """
#         Returns a list of objects detected in the image
#         """
#         pass

class ObjectDetectionModel(ABC):

    def train(self, images: List, labels: List[str]):
        """
        Trains the model on a list of images and their corresponding labels
        """
        pass

    @abstractmethod
    def detect(self, image: Image) -> List[DetectionResult]:
        """
        Returns a list of objects detected in the image
        """
        pass