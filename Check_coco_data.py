import os, json

with open('/Users/lewishall/Desktop/Masters Project Data Sets/Annas_dataset/coco/train/train_annotations.json') as f:
    data = json.load(f)

image_dir = '/Users/lewishall/Desktop/Masters Project Data Sets/Annas_dataset/coco/train/images/'

missing = [
    img['file_name'] for img in data['images']
    if not os.path.isfile(os.path.join(image_dir, img['file_name']))
]

print(f'Missing images: {len(missing)}')
