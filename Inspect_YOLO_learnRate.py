
# Activate the Miniconda environment 

# conda activate yolo_v8_env_1

# Navigate to directory where Train_YOLOv8.py is located.

# cd ~ Desktop

# Run training script

# python Inspect_YOLO_learnRate.py

import os
import ultralytics
from ultralytics import YOLO

model_path = os.path.expanduser("~/Desktop/yolov8s.yaml") # .pt or .yaml file
model = YOLO(model_path)  
print(model.args)

print("Training arguments:")
for k, v in model.args.items():
    print(f"{k}: {v}")