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

# Activate the Miniconda environment 

# conda activate yolo_v8_env_1

# Navigate to directory where Train_YOLOv8.py is located.

# cd ~ Desktop

# Run training script

# python Train_YOLOv8.py




import os
from ultralytics import YOLO




####### UPDATE Save directory #######
save_directory = os.path.expanduser("~/Desktop/YOLOv8_Models+Results/Model_9")  

# If you want to train a model using an existing model you already have then 
# Set the model path for the existing model 




####### UPDATE Pretrained model path #######
#model_path = os.path.expanduser("~/Desktop/YOLOv8_Models+Results/Model_5/Plastic_identifier_YOLOv8_5/weights/best.pt") 

# Create the YOLO model instance
#model = YOLO(model_path)

# To use a pretrained model set the model path to this 
model = YOLO('yolov8s.pt') # Load pretrained YOLOv8 model




# Path to the YAML file used for your training data

####### UPDATE YAML file path #######
data_config_path = os.path.expanduser("~/Desktop/Masters Project Data Sets/9_Final_training+Val_sets/9_Train+Val_7/9_Train+Val_7_YAML.yaml") 



####### Optional: epochs, imgzs, batch #######
# Start training
results = model.train(
    data=data_config_path, 
    epochs=50, 
    imgsz=640, 
    batch=16, 
    device='mps', 
    project=save_directory,  # Save to desktop

    ############ Model name ############
    name="Plastic_identifier_YOLOv8_9"  # CHANGE NAME 
)

print(f"Training completed! Model saved to: {save_directory}")
