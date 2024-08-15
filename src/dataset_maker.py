import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

import os
import json

def show_dataset(sample_path):
    """
    Show the dataset.
    """
    segmentation_path = f"{sample_path}/Scene.png"
    segmentation = Image.open(segmentation_path)

    fig, ax = plt.subplots()
    ax.imshow(segmentation)

    for type in os.listdir(f"{sample_path}/ground_truth"):
        for i in os.listdir(f"{sample_path}/ground_truth/{type}"):
            box = f"{sample_path}/ground_truth/{type}/{i}/bounding_box.jsonl"

            print(type, i, box)
            with open(box, "r") as f:
                bounding_box = json.loads(f.read())

                # Show the bounding box
                x_min = bounding_box["xmin"]
                y_min = bounding_box["ymin"]
                x_max = bounding_box["xmax"]
                y_max = bounding_box["ymax"]

                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
    plt.show()
    return segmentation

show_dataset("data/cityenviron/aerial/normal/train/images/22")