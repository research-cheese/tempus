import shutil
import os

def flatten_dataset(dataset_path, output_path):
    # If output path doesn't exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for sample in os.listdir(dataset_path):
        segmentation_sample_path = f"{dataset_path}/{sample}/Scene.png"
        segmentation_output_path = f"{output_path}/{sample}.png"

        shutil.copyfile(segmentation_sample_path, segmentation_output_path)


flatten_dataset("test/data/cityenviron/aerial/dust-0.5/train/images", "flattened/dust-0.5/train/images")
flatten_dataset("test/data/cityenviron/aerial/dust-0.5/test/images", "flattened/dust-0.5/test/images")

flatten_dataset("test/data/cityenviron/aerial/fog-0.5/train/images", "flattened/fog-0.5/train/images")
flatten_dataset("test/data/cityenviron/aerial/fog-0.5/test/images", "flattened/fog-0.5/test/images")

flatten_dataset("test/data/cityenviron/aerial/maple_leaf-0.5/train/images", "flattened/maple_leaf-0.5/train/images")
flatten_dataset("test/data/cityenviron/aerial/maple_leaf-0.5/test/images", "flattened/maple_leaf-0.5/test/images")

flatten_dataset("test/data/cityenviron/aerial/normal/train/images", "flattened/normal/train/images")
flatten_dataset("test/data/cityenviron/aerial/normal/test/images", "flattened/normal/test/images")

flatten_dataset("test/data/cityenviron/aerial/rain-0.5/train/images", "flattened/rain-0.5/train/images")
flatten_dataset("test/data/cityenviron/aerial/rain-0.5/test/images", "flattened/rain-0.5/test/images")

flatten_dataset("test/data/cityenviron/aerial/snow-0.5/train/images", "flattened/snow-0.5/train/images")
flatten_dataset("test/data/cityenviron/aerial/snow-0.5/test/images", "flattened/snow-0.5/test/images")