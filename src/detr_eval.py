import os

from utils.object_detection.detr.convert_to_coco_detection import convert_to_coco_detection_folder
from utils.object_detection.detr.detr_model import CocoDetection
from utils.core.env_var import Environment

from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection
import torch
from pytorch_lightning import Trainer
from tqdm.notebook import tqdm
from PIL import Image

CATEGORIES = ["pedestrian", "vehicle", "construction", "nature"]
ID_TO_LABEL = {i: label for i, label in enumerate(CATEGORIES)}

class MyDataLoader(DataLoader):
    def __init__(self, dataset, collate_fn, batch_size, shuffle=False):
        super().__init__(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)

    def __code__(self):
        return self.dataset.__code__
    
class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, train_dat = None, val_dat = None):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                                            num_labels=len(ID_TO_LABEL),
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.train_dat = train_dat
        self.val_dat = val_dat

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                    weight_decay=self.weight_decay)
        
        return optimizer

    def train_dataloader(self):
        return self.train_dat
    
    def val_dataloader(self):
        return self.val_dat

def get_coco_path(name):
    return f"coco_detection/cityenviron/aerial/{name}"
import torch
import matplotlib.pyplot as plt

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{ID_TO_LABEL[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
def visualize_predictions(image, outputs, threshold=0.9, keep_highest_scoring_bbox=False):
  # keep only predictions with confidence >= threshold
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  if keep_highest_scoring_bbox:
    keep = probas.max(-1).values.argmax()
    keep = torch.tensor([keep])
  
  # convert predicted boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    
  # plot results
  plot_results(image, probas[keep], bboxes_scaled)

def convert(name):
    convert_to_coco_detection_folder(
        dataset_folder_path=f"data/cityenviron/aerial/{name}",
        output_folder_path=get_coco_path(name),
        categories=CATEGORIES
    )

def eval(name):
    checkpoint_path = f"model_checkpoint/detr/{name}.ckpt"
    test_folder_path = f"coco_detection/cityenviron/aerial/{name}/test"
    test_folder_path = f"coco_detection/cityenviron/aerial/{name}/train"
    test_image_folder_path = f"{test_folder_path}/images"
    val_name = f"{name}/test"
    train_name = f"{name}/train"
    feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = model.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    model = Detr.load_from_checkpoint(checkpoint_path, strict=False, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
    model.eval()
    model.to(Environment().device)

    val_dataset = CocoDetection(dataset_folder=get_coco_path(train_name), feature_extractor=feature_extractor)
    it = iter(range(1500))
    pixel_values, target = val_dataset[next(it)]
    pixel_values = pixel_values.unsqueeze(0).to(Environment().device)
    outputs = model(pixel_values=pixel_values, pixel_mask=None)
    image_id = target['image_id'].item()
    image = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(f'{test_image_folder_path}', image['file_name']))

    visualize_predictions(image, outputs, threshold=0.3, keep_highest_scoring_bbox=True)


eval("dust-0.5")
eval("fog-0.5")
eval("maple_leaf-0.5")
eval("normal")
eval("rain-0.5")
eval("snow-0.5")