import os
import json

from PIL import Image
import shutil
from typing import List

def create_coco_annotation(id, image_id, bbox, area, category, categories_dict):
    """
    Creates a COCO annotation dictionary.
    """

    category_id = categories_dict[category]

    return {
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "bbox": bbox,
        "area": area
    }

def create_coco_image(id, file_name, width, height):
    """
    Creates a COCO image dictionary.
    """

    return {
        "id": id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def convert_to_coco_detection_folder(
    dataset_folder_path: str,
    output_folder_path: str,
    categories: List[str]
):
    """
    Converts images and annotations in the folders to COCO format and saves them in the output folder.
    """

    sorted_categories = sorted(categories)
    categories_dict = {category: i for i, category in enumerate(sorted_categories)}

    annotations = []
    images = []
    categories = []

    # ========================
    # Create categories array
    # ========================
    for category in sorted_categories:
        categories.append({
            "id": categories_dict[category],
            "name": category,
            "supercategory": category
        })

    # ========================
    # Create images and annotations arrays
    # ========================
    images_folder_path = f"{dataset_folder_path}/images"
    target_images_folder_path = f"{output_folder_path}/images"

    # If the output folder exists delete it
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(target_images_folder_path, exist_ok=True)
    for sample in os.listdir(images_folder_path):
        image_path = f"{images_folder_path}/{sample}/Scene.png"
        target_image_path = f"{output_folder_path}/images/{sample}.png"
        bbox_path = f"{images_folder_path}/{sample}/bounding_box.jsonl"

        # Copy image to output folder ==================
        shutil.copy(image_path, target_image_path)

        # Populate images array ========================
        image = Image.open(image_path)
        width, height = image.size
        image_id = len(images)
        images.append(create_coco_image(
            id=image_id,
            file_name=f"{sample}.png",
            width=width,
            height=height
        ))

        if not os.path.exists(bbox_path): continue
        # Populate annotations array ===================
        with open(bbox_path, "r") as f:
            for i, line in enumerate(f):
                # Files are stored in this format {"class": "vehicle", "xmin": 170, "ymin": 30, "xmax": 186, "ymax": 45} in each line
                bbox = eval(line)
                category = bbox["class"]
                width = bbox["xmax"] - bbox["xmin"]
                height = bbox["ymax"] - bbox["ymin"]
                area = width * height
                bbox = [bbox["xmin"], bbox["ymin"], width, height]

                annotations.append(create_coco_annotation(
                    id=i,
                    image_id=image_id,
                    bbox=bbox,
                    area=area,
                    category=category,
                    categories_dict=categories_dict
                ))

    with open(os.path.join(output_folder_path, "annotation.json"), "w") as f:
        f.write(json.dumps({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }))
