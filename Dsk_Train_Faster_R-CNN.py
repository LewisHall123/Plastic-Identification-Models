import os
from mmengine.config import Config
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

# Get username for Windows Desktop path
username = os.getlogin()

# === Paths ===
dataset_root = os.path.join(os.getcwd(), 'annas_coco')
train_img_dir = os.path.join(dataset_root, 'train', 'images')
train_ann_file = os.path.join(dataset_root, 'train', 'train_annotations.json')

val_img_dir = train_img_dir  # using same images folder
val_ann_file = os.path.join(dataset_root, 'train', 'val_annotations.json')

# === Load and modify base config ===
config_file = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
cfg = Config.fromfile(config_file)

cfg.dataset_type = 'CocoDataset'
cfg.data_root = dataset_root

cfg.metainfo = {'classes': ('PET', 'PP')}

cfg.train_dataloader.dataset.ann_file = train_ann_file
cfg.train_dataloader.dataset.img_prefix = train_img_dir
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = val_ann_file
cfg.val_dataloader.dataset.img_prefix = val_img_dir
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader

# === Model settings ===
cfg.model.roi_head.bbox_head.num_classes = 2  # PET, PP
cfg.device = 'cuda'

# === Work dir (save model to Desktop) ===
output_path = os.path.join('C:\\Users', username, 'Desktop', 'faster_rcnn_output')
cfg.work_dir = output_path
os.makedirs(output_path, exist_ok=True)

# === Training config tweaks ===
cfg.train_cfg.max_epochs = 10  # or higher if needed
cfg.default_hooks.logger.interval = 10
cfg.default_hooks.checkpoint.interval = 1
cfg.randomness = dict(seed=42, deterministic=False)

# === Build model and dataset ===
model = build_detector(cfg.model)
model.init_weights()
datasets = [build_dataset(cfg.train_dataloader.dataset)]

# === Start training ===
train_detector(model, datasets, cfg, distributed=False, validate=True)

print(f"\nTraining done. Model saved to: {os.path.join(output_path, 'latest.pth')}")
