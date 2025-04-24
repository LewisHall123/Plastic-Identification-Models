
###### There are notes at the bottom of this page

##########################################
# How to test a model
##########################################

# Read through the testing code, update it where necessary, save it.
    # Update path to model
    # Update path for each test set
    # Update path to output folder  
    # SAVE THIS FILE

# Activate the Miniconda environment 

# conda activate yolo_v8_env_1

# Navigate to directory where Test_YOLOv8.py is located.

# cd ~ Desktop

# Run testing script

# python Test_YOLOv8.py


import os
from ultralytics import YOLO

####### UPDATE PATH to model #######
model_path = os.path.expanduser("~/Desktop/YOLOv12_Models+Results/Model_12/Plastic_identifier_YOLOv12_12/weights/best.pt")  

# To test individually on one test dataset you can use this code:

####### UPDATE PATH to test dataset(s)#######
# test_images_path = "path/to/your/test/images"  # Update this path

# To test model on multiple test datasets 
# Define test datasets (keys = dataset names, values = paths to test images)

test_datasets = { 

    ####### UPDATE path for each test set #######
    # "Test_scenario_1": "/Users/lewishall/Desktop/Masters Project Data Sets/Annas_dataset/yolo/data.yaml",
    # "Test_scenario_2": "/Users/lewishall/Desktop/Masters Project Data Sets/5_Test_scenarios/Test_scenario_2/yolo/5_Test_2_YAML.yaml",
    # "Test_scenario_3": "/Users/lewishall/Desktop/Masters Project Data Sets/5_Test_scenarios/Test_scenario_3/yolo/5_Test_3_YAML.yaml",
    "Test_scenario_4": "/Users/lewishall/Desktop/Masters Project Data Sets/5_Test_scenarios/Test_scenario_4/yolo/5_Test_4_YAML.yaml"
    #"Test_scenario_5": "/Users/lewishall/Desktop/Masters Project Data Sets/5_Test_scenarios/Test_scenario_4/yolo/images"
}

# Load the trained model
model = YOLO(model_path)

# Define output folder on Desktop

####### UPDATE PATH to output folder #######
output_base_dir = os.path.expanduser("~/Desktop/YOLOv12_Models+Results/Model_12_Results") 


########## Basic Testing ##########
'''for name, yaml_path in test_datasets.items():
    output_dir = os.path.join(output_base_dir, name)  # Unique folder per dataset
    
    results = model.val(
        data=yaml_path,
        imgsz=640,
        device="mps",
        project=output_base_dir,
        name=name
    )'''


########## Testing with metrics ##########
for name, yaml_path in test_datasets.items():
    output_dir = os.path.join(output_base_dir, name)  # Unique folder per dataset

    results = model.val(
        data=yaml_path,
        imgsz=640,
        device="mps",
        project=output_base_dir,
        name=name
    )

    # Get mean metrics using methods
    mean_precision = results.box.mp       # Mean Precision
    mean_recall = results.box.mr          # Mean Recall
    map50 = results.box.map50             # mAP@0.5
    map50_95 = results.box.map            # mAP@0.5:0.95

    # Extract speed metrics
    speed = results.speed  # Dictionary with 'preprocess', 'inference', 'loss', 'postprocess'

    # Start formatting the output string
    result_text = (
        f"{name} Results:\n"
        f"{'-'*40}\n"
        f"Mean Precision: {mean_precision:.4f}\n"
        f"Mean Recall:    {mean_recall:.4f}\n"
        f"mAP@0.5:        {map50:.4f}\n"
        f"mAP@0.5:0.95:   {map50_95:.4f}\n\n"
        f"Speed (ms per image):\n"
        f"  Preprocess:   {speed['preprocess']:.2f}\n"
        f"  Inference:    {speed['inference']:.2f}\n"
        f"  Postprocess:  {speed['postprocess']:.2f}\n"
        f"{'-'*40}\n"
        f"Per-Class Metrics:\n"
    )

    # Per-class metrics
    for i, class_name in model.names.items():
        try:
            p, r, ap50, ap = results.box.class_result(i)
            result_text += (
                f"  {class_name}:\n"
                f"    Precision:     {p:.4f}\n"
                f"    Recall:        {r:.4f}\n"
                f"    mAP@0.5:       {ap50:.4f}\n"
                f"    mAP@0.5:0.95:  {ap:.4f}\n"
            )
        except IndexError:
            result_text += (
                f"  {class_name}:\n"
                f"    No instances in test set.\n"
            )

    # Print to terminal
    print("\n" + result_text)

    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "metrics.txt")
    with open(results_file, "w") as f:
        f.write(result_text)



################## inference ##################
'''
# Loop through each test dataset and run inference
for dataset_name, test_images_path in test_datasets.items():
    output_dir = os.path.join(output_base_dir, dataset_name)  # Unique folder per dataset

    # Run inference and save results
    # Running inference in the context of YOLOv8 means using a trained 
    # model to make predictions on new, unseen images or videos.
    results = model.predict(
        source=test_images_path,
        imgsz=640,
        device="mps",
        save=True,
        project=output_dir,  
        name=dataset_name  # Creates separate folder for each dataset
    )

    print(f"Inference completed for {dataset_name}! Results saved in: {output_dir}")
'''


# By default, YOLOv8 automatically saves all training and inference results inside the 
# runs/ directory, which is created in your current working directory (i.e., wherever
# you're running the script from).

# If you don’t specify a project= argument in the training or testing script,
# YOLOv8 saves results in:

# YOLOv8 saves results in: YOUR_CURRENT_DIRECTORY/runs/
# So, if you run the script from ~/Documents/YOLO_Project/, you'll find:
# ~/Documents/YOLO_Project/runs/
# runs/train/ → Stores training logs & weights.
# runs/detect/ → Stores inference results (test/predictions).

# For the path to model:
# It ends with /weights/best.pt because YOLOv8 saves the best-performing model 
# checkpoint inside the weights/ directory during training. /weights is a subfolder to store 
# different model weight checkpoints. best.pt is the best model based on validation performance
# last.pt is the final model checkpoint after all training epochs.