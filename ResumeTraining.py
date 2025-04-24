########## How to use ##########
# Change model to continue training
# SAVE this file
# Use terminal to run the script


import os
from ultralytics import YOLO

########### Change model to continue training ###########
last_checkpoint_of_model = os.path.expanduser("~/Desktop/YOLOv12_Models+Results/Model_12/Plastic_identifier_YOLOv12_12/weights/last.pt")

# Load your saved model (this should point to the 'last.pt' checkpoint by default)
model = YOLO(last_checkpoint_of_model)  

# Resume training
model.train(resume=True, device="mps")  # Use 'mps' for Mac with Apple silicon, or 'cuda' for NVIDIA GPUs

########## TERMINAL COMMAND ##########

# python ResumeTraining.py