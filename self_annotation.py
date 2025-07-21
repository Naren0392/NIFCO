import os
import cv2
import numpy as np

# Input image folder
img_dir = r"C:\Users\Admin\Downloads\PartImages (1)\3_26812&13"
label_dir = os.path.join(img_dir, "labels")
annotated_dir = os.path.join(img_dir, "annotated")

os.makedirs(label_dir, exist_ok=True)
os.makedirs(annotated_dir, exist_ok=True)

for img_file in os.listdir(img_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 250  # threshold: tweak if needed
    coords = cv2.findNonZero(mask.astype(np.uint8))

    if coords is None:
        continue

    x, y, w, h = cv2.boundingRect(coords)

    H, W = img.shape[:2]
    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    w_norm = w / W
    h_norm = h / H

    # Save YOLO label file
    txt_name = os.path.splitext(img_file)[0] + ".txt"
    txt_path = os.path.join(label_dir, txt_name)
    with open(txt_path, "w") as f:
        f.write(f"3 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Draw bounding box on image
    annotated_img = img.copy()
    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(annotated_img, "26812&13", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save annotated image
    out_path = os.path.join(annotated_dir, img_file)
    cv2.imwrite(out_path, annotated_img)

print("✅ Labels and annotated images saved.")
