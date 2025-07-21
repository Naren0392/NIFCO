# import json
# from ultralytics import YOLO
# from multiprocessing import freeze_support
# import torch

# if __name__ == '__main__':
#     freeze_support()

#     print("CUDA Available:", torch.cuda.is_available())

#     # Load config
#     with open(r"D:/Vision Analytics/Conc dataset PPE/NIFCO/config_file.json") as data:
#         config = json.load(data)

#     # Load model
#     model = YOLO("yolov8m.pt")  # You can replace this with your custom pretrained model if any
#     model.info()

#     # Train model with image size 640
#     results = model.train(
#         data=config['yaml_file_path'],      # Path to your YAML data config
#         epochs=50,                          # Number of epochs
#         imgsz=640,                          # Set image size to 640
#         project=r"D:\Vision Analytics\Conc dataset PPE\NIFCO\03-07_training",
#         device=0                            # Use GPU 0
    # )

import os
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support  # Required for Windows multiprocessing

def main():
    # ✅ Check CUDA (GPU) availability
    if torch.cuda.is_available():
        device = "0"  # Use CUDA GPU device 0
        print(f"[INFO] CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"  # Fallback to CPU
        print("[WARNING] CUDA not available. Using CPU instead.")

    # Set dataset paths
    train_images = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\New folder\train"
    val_images = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\New folder\valid"
    test_images = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\New folder\test"


    # Set number of classes and class names
    num_classes = 2
    class_names = [f"class{i}" for i in range(num_classes)]

    # Convert list to YAML format
    class_names_str = "[" + ", ".join([f'"{name}"' for name in class_names]) + "]"

    # Create YAML content
    yaml_content = (
        f"train: {train_images}\n"
        f"val: {val_images}\n"
        f"test: {test_images}\n"
        f"nc: {num_classes}\n"
        f"names: {class_names_str}\n"
    )

    # Write to file
    yaml_path = "partimages.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[INFO] data.yaml written to {yaml_path}")

    # Load YOLOv8n model
    model = YOLO("yolov8n.pt")  # You can change this to yolov8s.pt, yolov8m.pt, etc.

    # Train the model
    # model.train(
    #     data=yaml_path,
    #     epochs=50,
    #     imgsz=640,
    #     batch=16,
    #     name="yolov_train_partimages",
    #     device=device  # Automatically use CUDA or CPU based on availability
    # )
    model.train(
    data=yaml_path,
    epochs=65,
    imgsz=640,
    batch=16,
    name="yolov_train_partimages",
    device=device,
    optimizer='Adam'
)

if __name__ == '__main__':
    freeze_support()  # Required on Windows for multiprocessing
    main()
