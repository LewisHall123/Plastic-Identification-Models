from mmengine.config import Config
from mmengine.runner import Runner
import os

# === Paths (Google Drive Mounted) ===
drive_root = '/content/drive/MyDrive'
dataset_root = os.path.join(drive_root, '7_coco')

train_ann = os.path.join(dataset_root, 'train', 'train_annotations.json')
val_ann = os.path.join(dataset_root, 'train', 'val_annotations.json')
train_img = os.path.join(dataset_root, 'train', 'images')
val_img = train_img  # using same image folder

# === Load base config ===
cfg = Config.fromfile('configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')

cfg.dataset_type = 'CocoDataset'
cfg.data_root = dataset_root
cfg.metainfo = {'classes': ('PET', 'PP', 'HDPE')}

# === Train dataset ===
cfg.train_dataloader.dataset = dict(
    type='CocoDataset',
    data_root=dataset_root,
    ann_file=train_ann,
    data_prefix=dict(img='train/images'),
    metainfo=cfg.metainfo,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=cfg.train_dataloader.dataset.pipeline
)

# === Validation dataset ===
cfg.val_dataloader.dataset = dict(
    type='CocoDataset',
    data_root=dataset_root,
    ann_file=val_ann,
    data_prefix=dict(img='train/images'),
    metainfo=cfg.metainfo,
    test_mode=True,
    pipeline=cfg.val_dataloader.dataset.pipeline
)

# === Also use val set for test_dataloader ===
cfg.test_dataloader = cfg.val_dataloader

# === Model ===
cfg.model.roi_head.bbox_head.num_classes = 3
cfg.device = 'cuda'

# === Output ===
cfg.work_dir = os.path.join(drive_root, 'faster_rcnn_output_2')
os.makedirs(cfg.work_dir, exist_ok=True)

# === Train loop config (with validation every epoch) ===
cfg.train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1
)

# === Hooks and randomness ===
cfg.default_hooks.checkpoint.interval = 1
cfg.default_hooks.logger.interval = 10
cfg.randomness = dict(seed=42, deterministic=False)

# === Evaluators ===
cfg.val_evaluator = dict(
    type='CocoEvaluator',
    ann_file=val_ann,
    metric='bbox'
)
cfg.test_evaluator = cfg.val_evaluator

# === Learning rate scheduler ===
cfg.param_scheduler = [
    dict(
        type='MultiStepLR',
        milestones=[7, 9],
        gamma=0.1,
        by_epoch=True
    )
]

# === Early stopping hook ===
cfg.default_hooks.early_stop = dict(
    type='EarlyStoppingHook',
    monitor='coco/bbox_mAP',
    rule='greater',
    patience=3
)

# === Fix evaluator annotation file paths ===
cfg.val_evaluator.ann_file = val_ann
cfg.test_evaluator.ann_file = val_ann

# === Start training ===
runner = Runner.from_cfg(cfg)
runner.train()