from pycocotools.coco import COCO

coco = COCO('/Users/lewishall/Desktop/Masters Project Data Sets/Annas_dataset/coco/train/train_annotations.json')
img_ids = coco.getImgIds()
valid_img_ids = [img_id for img_id in img_ids if len(coco.getAnnIds(img_id)) > 0]

print(f"Images with at least one annotation: {len(valid_img_ids)}")