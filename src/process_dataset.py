import os
import shutil
import json

from utils.preprocessing.get_ground_truth import get_ground_truth
from utils.core.hex_to_rgb import hex_to_rgb
from utils.preprocessing.cityenviron_colors import HUMAN_COLORS, VEHICLE_COLORS, CONSTRUCTION_COLORS, NATURE_COLORS
from utils.preprocessing.save_black_and_white_image import save_black_and_white_image

import matplotlib.pyplot as plt
from PIL import Image

def process_dataset(dataset_path):
    image_path = f"{dataset_path}/images"
    for sample in os.listdir(image_path):
        print(f"Processing {sample}")
        sample_path = f"{image_path}/{sample}"
        segmentation_path = f"{sample_path}/Segmentation.png"                

        ground_truth_path = f"{sample_path}/ground_truth"
        if os.path.exists(ground_truth_path): shutil.rmtree(ground_truth_path)

        saved_bounding_box_path = f"{sample_path}/bounding_box.jsonl"
        if os.path.exists(saved_bounding_box_path): os.remove(saved_bounding_box_path)
        os.makedirs(ground_truth_path)

        for colors, color_name, group_disjoint, min_count in [
            (HUMAN_COLORS, "pedestrian", False, 10), 
            (VEHICLE_COLORS, "vehicle", False, 30), 
            (CONSTRUCTION_COLORS, "construction", True, 30), 
            (NATURE_COLORS, "nature", False, 50)]:
            truth = get_ground_truth(segmentation_path, colors, group_disjoint=group_disjoint, min_count=min_count)

            index = 0
            for t in truth:
                ground_truth_sample_path = f"{sample_path}/ground_truth/{color_name}/{index}"
                os.makedirs(ground_truth_sample_path)
                
                saved_segmentation_path = f"{ground_truth_sample_path}/segmentation.png"
                save_black_and_white_image(t.segmentation, saved_segmentation_path)

                with open(saved_bounding_box_path, "a") as f:
                    f.write(json.dumps({"class": color_name, **t.bounding_box.__dict__}))
                    f.write("\n")

                index += 1

# process_dataset("data/cityenviron/aerial/dust-0.5/train")
# process_dataset("data/cityenviron/aerial/dust-0.5/test")
# process_dataset("data/cityenviron/aerial/fog-0.5/train")
# process_dataset("data/cityenviron/aerial/fog-0.5/test")
# process_dataset("data/cityenviron/aerial/maple_leaf-0.5/train")
# process_dataset("data/cityenviron/aerial/maple_leaf-0.5/test")
# process_dataset("data/cityenviron/aerial/normal/train")
# process_dataset("data/cityenviron/aerial/normal/test")
# process_dataset("data/cityenviron/aerial/rain-0.5/train")
# process_dataset("data/cityenviron/aerial/rain-0.5/test")
# process_dataset("data/cityenviron/aerial/snow-0.5/train")
# process_dataset("data/cityenviron/aerial/snow-0.5/test")
process_dataset("data/cityenviron/aerial/test")