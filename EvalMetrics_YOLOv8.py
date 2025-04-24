############################################################
# Code to find metrics using the results file
############################################################

import os
import pandas as pd
from ultralytics import YOLO 

# Paths
output_base_dir = os.path.expanduser("~/Desktop/YOLOv8_Models+Results/Model_3_Results/Annas_Test_Polygon/Dataset1")
results_metrics_path = os.path.expanduser("~/Desktop/YOLOv8_Models+Results/Model_3_Results/Annas_Test_Polygon/Results_Analytics/metrics.csv")

# Prepare a list to store metrics
metrics_list = []

# Loop through each dataset's output folder to calculate metrics
for dataset_name in os.listdir(output_base_dir):
    dataset_folder = os.path.join(output_base_dir, dataset_name)
    if os.path.isdir(dataset_folder):  # Check if it is a directory
        
        # Adjust to your actual model file name
        model_path = os.path.join(dataset_folder, 'best.pt')  
        
        # Load the results for this dataset
        results = YOLO(os.path.join(dataset_folder, 'predict'))  # Adjust path to where predictions are saved

        # Capture overall performance metrics (you might want to calculate these)
        overall_metrics = {
            "Dataset": dataset_name,
            "Precision": results.metrics['precision'],  # Modify according to available results
            "Recall": results.metrics['recall'],        # Modify according to available results
            "F1": results.metrics['f1'],                 # Modify according to available results
        }
        metrics_list.append(overall_metrics)

# Save overall metrics to CSV at the specified path on your desktop
os.makedirs(os.path.dirname(results_metrics_path), exist_ok=True)  # Create directory if it doesn't exist
pd.DataFrame(metrics_list).to_csv(results_metrics_path, index=False)

print(f"Metrics calculated and saved in: {results_metrics_path}")
