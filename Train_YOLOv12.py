##########################################
# How to trian a model
##########################################

# Read through the training code, update it where necessary, save it.
    # Update Save directory path
    # Update Pretrained model path or load a pretrained model
    # Update Yaml file path
    # Optional: epochs, imgzs, batch 
    # Model name 
    # SAVE

# Make sure you have the YOLOv12 pretrained model downloaded onto your desktop before you start training. Here's the link to find it:
# https://github.com/sunsmarterjie/yolov12?tab=readme-ov-file

# Activate the Miniconda environment 

# conda activate yolov12_env_2

# Navigate to directory where Train_YOLOv12.py is located.

# cd ~ Desktop
# cd ~ Masters Project Code

# Run training script

# python Train_YOLOv12.py




import os
from ultralytics import YOLO




####### UPDATE Save directory #######
save_directory = os.path.expanduser("~/Desktop/YOLOv12_Models+Results/Model_12")  

# If you want to train a model using an existing model you already have then 
# Set the model path for the existing model 




####### UPDATE Pretrained model path #######
#model_path = os.path.expanduser("~/Desktop/YOLOv8_Models+Results/Model_5/Plastic_identifier_YOLOv8_5/weights/best.pt") 

# Create the YOLO model instance
#model = YOLO(model_path)

# To use a pretrained model set the model path to this 
model = YOLO('yolov12s.pt') # Load pretrained YOLOv8 model




# Path to the YAML file used for your training data

####### UPDATE YAML file path #######
data_config_path = os.path.expanduser("~/Desktop/Masters Project Data Sets/9_Final_training+Val_sets/9_Train+Val_7/9_Train+Val_7_YAML.yaml") 




# Start training

####### UPDATE epochs, imgzs, batch  #######

results = model.train(
    data=data_config_path, 
    epochs=50, 
    imgsz=640, 
    batch=16, 
    device='mps', 
    project=save_directory,  # Save to desktop

    ##################### UPDATE name  #####################
    name="Plastic_identifier_YOLOv12_12",  
    verbose=True,  # Show training progress in detail
)

print(f"Training completed! Model saved to: {save_directory}")
