from utils.airsim.computer_vision_session import ComputerVisionSession

import airsim

ms = ComputerVisionSession()

camera_name = "low_res"
image_type = airsim.ImageType.Scene

ms.client.simSetDetectionFilterRadius(camera_name, image_type, 80 * 100)
ms.client.simAddDetectionFilterMeshName(camera_name, image_type, "*")
detections = ms.client.simGetDetections(camera_name, image_type)
print(detections)