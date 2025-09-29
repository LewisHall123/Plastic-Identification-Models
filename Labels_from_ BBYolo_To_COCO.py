
# This code changes the format of the bounding box labels from regular YOLO bounding box to COCO format. 


############# How to use ##############
# Change paths to images, labels, and output JSON file



import os
import json
#  From the Pillow library — used to load each image and retrieve its width and height. 
# This is essential to convert normalised YOLO coordinates into absolute pixel coordinates, which COCO format uses.
from PIL import Image 

# os.path.expanduser("~/Desktop/Masters Project Data Sets/9_Final_training+Val_sets/9_Train+Val_2/coco/coco_train")
# yolo/train/labels
# yolo/valid/labels
# 5_Test_scenarios/Test_scenario_1/yolo/labels
# 9_Final_training+Val_sets/9_Train+Val_7/train/images
# 5_Test_scenarios/Test_scenario_4/yolo/labels

############ Change paths to images, labels, output JSON file, and name of output file from test to train etc ############

# The folder containing your .jpg images.
image_dir = os.path.expanduser("~/Desktop/Masters Project Data Sets/5_Test_scenarios/Test_scenario_4/yolo/images") 

# The folder containing your YOLO-format .txt files — same name as the image but different extension.
label_dir = os.path.expanduser("~/Desktop/Masters Project Data Sets/5_Test_scenarios/Test_scenario_4/yolo/labels") 

# The name of the final .json file to be saved. Change 
output_json = os.path.expanduser("~/Desktop/Masters Project Data Sets/5_Test_scenarios/Test_scenario_4/coco/test_annotations.json")   
classes = ["PET", "PP", "HDPE"]           # Keep the order consistent with YOLO class IDs


# COCO structures
images = []             # A list of metadata for each image (filename, size, ID).
annotations = []        # A list of objects (bounding boxes) detected in those images.
categories = []         # A list of your object classes for COCO format.
annotation_id = 0       # A unique ID given to each bounding box annotation.
image_id = 0            # A unique ID for each image.


# Create category mapping
for i, class_name in enumerate(classes):
    categories.append({
        "id": i,
        "name": class_name,         # class name (e.g., "PET"),
        # a broader category that groups the classes (useful for grouping by type — here, all were "plastic", 
        # but annas data uses class name so I will use the same).
        "supercategory": class_name,  
    })

# Iterate through images
for filename in os.listdir(image_dir):  # Loops through all files in the image folder.
    if not filename.endswith(".jpg"):   # Skips any file that is not a .jpg.
        continue

    # Builds the full file path to each .jpg image and the corresponding YOLO .txt annotation file (which must have the same base filename).
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

    # Opens the image to read its width and height in pixels. These are needed to convert YOLO normalised coordinates (0 to 1) into absolute pixel values for COCO.

    # Using 'with' ensures that the file is properly closed after its suite finishes, even if an exception is raised.
    # Image.open() loads an image file from the given path.

    # It does not immediately read the full image into memory — it opens a lightweight image object that can be processed.
    # img becomes a PIL Image object, which gives you access to image attributes like .size, .mode, .format, etc.

    with Image.open(image_path) as img: 
        width, height = img.size # .size returns a tuple: (width, height) in pixels.

    # Adds an image record as a dictionary to the COCO images list, making the dictionary be an element of the list:
    images.append({
        "id": image_id,         # "id": Unique identifier for the image.
        "file_name": filename,  # "file_name": The filename (e.g., "img001.jpg").
        "width": width,         # "width" and "height": Dimensions used to scale YOLO coordinates.
        "height": height
    })

    # Read annotations
    with open(label_path, "r") as f:            # Opens the YOLO .txt file and reads all lines.
        for line in f.readlines():              # Each line corresponds to one bounding box in YOLO format.

            # map applies the float function to each element of the line split by spaces.
            # The split() method splits the string into a list of substrings based on whitespace.

            class_id, x_c, y_c, w, h = map(float, line.strip().split()) # class_id: Index of the object class (0, 1, or 2).
            x = (x_c - w / 2) * width
            y = (y_c - h / 2) * height
            w *= width                          # Multiplying the normalised width of the bounding box by the image width to get the absolute pixel width.
            h *= height

            annotations.append({
                "id": annotation_id,            # Unique ID of the annotation.
                "image_id": image_id,           # ID of the image this annotation belongs to.
                "category_id": int(class_id),   # Class index matching what's in the categories list.
                "bbox": [x, y, w, h],           # Pixel values.
                "area": w * h,                  # Total area of the bounding box (used by COCO metrics).
                "iscrowd": 0                    # Set to 0 (means object is not part of a crowd).
            })
            annotation_id += 1                  # Increment the annotation ID for the next bounding box.

    image_id += 1                               # Increment the image ID for the next image.

# Final COCO structure, builds the final dictionary in COCO format, ready to be saved as .json.
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

# Save to JSON

# Without with, you’d have to manually call f.close() after writing, and forgetting to close can lead to: File corruption, File locks, Memory leaks, Data loss.
# f is just a variable name that now refers to the open file object.
with open(output_json, "w") as f:

    # json.dump serialises (i.e. converts) a Python object (e.g., dictionary, list) into JSON format
    # Writes that serialised JSON string directly into the file object f
    # This is different from json.dumps(...) (with an “s”), which returns the JSON as a string, not as a file write
    # indent=4 → tells Python to: Format the output with indentation of 4 spaces per level. Makes the JSON human-readable (pretty-printed). Without indent, the JSON would be all on one line (hard to read)
    json.dump(coco_output, f, indent=4)

print(f"Saved COCO annotations to {output_json}")
