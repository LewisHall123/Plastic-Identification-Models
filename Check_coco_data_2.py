from pycocotools.coco import COCO

ann_file = "/Users/lewishall/Desktop/Masters Project Data Sets/Annas_dataset/coco/train/train_annotations.json"
coco = COCO(ann_file)
img_ids = coco.getImgIds()
valid_img_count = 0

for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    if len(anns) > 0:
        valid_img_count += 1

print(f"\n {valid_img_count} images have at least one valid annotation.")