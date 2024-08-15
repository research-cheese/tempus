import os

import torchvision

from utils.object_detection.object_detection_model import ObjectDetectionModel

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, dataset_folder, feature_extractor):
        img_folder = os.path.join(dataset_folder, "images")
        ann_file = os.path.join(dataset_folder, "annotation.json")
        
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

class DetrModel(ObjectDetectionModel):
    def __init__(self):
        pass

    def train(self, dataset):
        pass

    def detect(self, image):
        pass