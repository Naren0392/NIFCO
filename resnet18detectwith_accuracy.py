# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# import cv2
# import os
# from glob import glob
# from PIL import Image

# # ----------- Configuration -----------
# # image_folder = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\1_26116&17\train\26117_LH"

# image_folder = r"D:\downloads_archieve_14_07_2025\26117"

# # image_folder = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\1_26116&17\train\26116_RH"
# # image_folder = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\Test"
# model_weights_path = r"D:\Vision Analytics\Conc dataset PPE\NIFCO\resnet18_lhs_rhs_classifier26116&17.pth"
# num_classes = 2
# class_names = ['26116', '26117']

# # ----------- Load Model -----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(weights=None)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
# model.load_state_dict(state_dict)
# model = model.to(device)
# model.eval()

# # ----------- Image Transform -----------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

# # ----------- Load Images -----------
# image_paths = glob(os.path.join(image_folder, "*.jpg")) + glob(os.path.join(image_folder, "*.png"))
# print(f" Found {len(image_paths)} images in: {image_folder}")

# if len(image_paths) == 0:
#     print(" No images found. Please check the folder and file extensions.")
#     exit()

# # ----------- Inference Loop -----------
# for img_path in image_paths:
#     print(f"🔍 Processing: {img_path}")
#     try:
#         img = Image.open(img_path).convert('RGB')
#         input_tensor = transform(img).unsqueeze(0).to(device)

#         with torch.no_grad():
#             outputs = model(input_tensor)
#             probs = torch.softmax(outputs, dim=1)
#             pred_idx = torch.argmax(probs, dim=1).item()
#             confidence = probs[0, pred_idx].item()
#             label = class_names[pred_idx]

#             print(f" Prediction: {label} | Confidence: {confidence:.2f}")

#         # OpenCV image (optional display)
#         img_cv2 = cv2.imread(img_path)
#         img_resized = cv2.resize(img_cv2, (640, 640))
#         display_text = f"{label} ({confidence*100:.1f}%)"
#         cv2.putText(img_resized, display_text, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow('Detection', img_resized)
#         key = cv2.waitKey(0)
#         if key == ord('q'):
#             print(" Quitting on user input.")
#             break

#     except Exception as e:
#         print(f" Error processing {img_path}: {e}")

# cv2.destroyAllWindows()
# print(" Inference complete.")


import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

# ----------- Configuration -----------
image_path = r"D:\downloads_archieve_14_07_2025\26116\20250617_105541039_0000000000_T0000_01_CAM.png"  # 🔁 Change this path per image
model_weights_path = r"D:\Vision Analytics\Conc dataset PPE\NIFCO\resnet18_lhs_rhs_classifier26116&17.pth"
num_classes = 2
class_names = ['26116', '26117']

# ----------- Load Model -----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ----------- Image Transform -----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ----------- Inference -----------
try:
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()
        label = class_names[pred_idx]

        print(f" Prediction: {label} | Confidence: {confidence:.2f}")

    # ----------- Optional: Display Image -----------
    img_cv2 = cv2.imread(image_path)
    img_resized = cv2.resize(img_cv2, (640, 640))
    display_text = f"{label} ({confidence * 100:.1f}%)"
    cv2.putText(img_resized, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Prediction', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"❌ Error processing image: {e}")
