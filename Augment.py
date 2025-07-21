import os
import cv2
import albumentations as A

# List of your image folders
folders = [
    # r"C:\Users\Narayanasamy\Downloads\PartImages\27734",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\27735",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\23435",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\train\23435",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\train\23436",
    r"D:\downloads_archieve_14_07_2025\26116"
   
    # r"C:\Users\Narayanasamy\Downloads\PartImages\23436",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\26116",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\26117",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\26812",
    # r"C:\Users\Narayanasamy\Downloads\PartImages\26813"
]

# Define augmentation pipeline
# augmentation = A.Compose([
#     A.OneOf([
#         A.GaussianBlur(blur_limit=(3, 7), p=0.8),
#         A.MotionBlur(blur_limit=5, p=0.5),
#     ], p=1.0),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
#     A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5),
# ])
augmentation = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(5, 10), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.8),
    ], p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.4, p=0.6),
    A.MultiplicativeNoise(multiplier=(0.5, 1.6), per_channel=True, p=0.3),
])

# Number of duplicates per image
num_augmented = 5

# Process each folder
for folder in folders:
    print(f"\nProcessing folder: {folder}")
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            base_name = os.path.splitext(filename)[0]

            # Generate 5 augmented versions
            for i in range(1, num_augmented + 1):
                augmented = augmentation(image=image)['image']
                new_filename = f"{base_name}_aug{i}.jpg"
                new_filepath = os.path.join(folder, new_filename)
                cv2.imwrite(new_filepath, augmented)
                print(f"Saved: {new_filepath}")
