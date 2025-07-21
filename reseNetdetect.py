import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import cv2

# === PATHS ===
model_path = r"D:\Vision Analytics\Conc dataset PPE\NIFCO\resnet18_lhs_rhs_classifier26116&17.pth"
# image_folder = r"C:\Users\Admin\Downloads\ast\New-New\test\26117"
image_folder = r"D:\downloads_archieve_14_07_2025\PartImages 5\Test"

# === Setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Class names (update if changed during training) ===
class_names = ['26116', '26117']

# === Load model ===
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Define transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Loop through all images in the folder ===
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(image_folder, filename)

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
                pred_class = class_names[pred_idx]

            # Load with OpenCV for display
            img_cv = cv2.imread(img_path)
            if img_cv is not None:
                label = f"{pred_class} ({confidence.item() * 100:.1f}%)"
                cv2.putText(img_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0,255 ), 4)

                # ✅ Resize image for display
                img_resized = cv2.resize(img_cv, (640, 640))

                cv2.imshow("Prediction", img_resized)
 
                print(f"{filename} -> {label}")
                key = cv2.waitKey(0)
                if key == 27:  # Esc to break
                    break

        except Exception as e:
            print(f"Error processing {filename}: {e}")

cv2.destroyAllWindows()
