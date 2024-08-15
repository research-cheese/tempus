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

CATEGORIES = ["pedestrian", "vehicle", "construction", "nature"]
ID_TO_LABEL = {i: label for i, label in enumerate(CATEGORIES)}

class MyDataLoader(DataLoader):
    def __init__(self, dataset, collate_fn, batch_size, shuffle=False):
        super().__init__(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)

    def __code__(self):
        return self.dataset.__code__

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, train_dat, val_dat):
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

def convert(name):
    convert_to_coco_detection_folder(
        dataset_folder_path=f"data/cityenviron/aerial/{name}",
        output_folder_path=get_coco_path(name),
        categories=CATEGORIES
    )

convert("dust-0.5/train")
convert("fog-0.5/train")
convert("maple_leaf-0.5/train")
convert("normal/train")
convert("rain-0.5/train")
convert("snow-0.5/train")

convert("dust-0.5/test")
convert("fog-0.5/test")
convert("maple_leaf-0.5/test")
convert("normal/test")
convert("rain-0.5/test")
convert("snow-0.5/test")

def train(name):
    train_name = f"{name}/train"
    val_name = f"{name}/test"

    feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    
    train_dataset = CocoDetection(dataset_folder=get_coco_path(train_name), feature_extractor=feature_extractor)
    val_dataset = CocoDetection(dataset_folder=get_coco_path(val_name), feature_extractor=feature_extractor)

    train_dataloader = MyDataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    val_dataloader = MyDataLoader(val_dataset, collate_fn=collate_fn, batch_size=4)

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train_dat=train_dataloader, val_dat=val_dataloader)
    model.eval()
    device = torch.device(Environment().device)

    model.to(device)

    trainer = Trainer(max_steps=100, gradient_clip_val=0.1)
    trainer.fit(model)

    trainer.save_checkpoint(f"model_checkpoint/detr/{name}.ckpt")

train("dust-0.5")