# import cv2
# import os
# from ultralytics import YOLO

# # Set model path and image folder
# model_path = r"D:\face_vsrun\runs\detect\yolov_train_partimages18\weights\best.pt"
# # image_folder = r"D:\downloads_archieve_14_07_2025\26116"
# # image_folder = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\1_26116&17\train\26117_LH"
# image_folder = r"D:\downloads_archieve_14_07_2025\26117"
# # image_folder = r"D:\downloads_archieve_14_07_2025\PartImages (1) - Copy\1_26116&17\train\26116_RH"

# # image_folder = r"D:\downloads_archieve_14_07_2025\26116"

# # Custom class names
# custom_names = {
#     0: "17&16",
#     1: "New part",
# }

# # Load the YOLO model
# model = YOLO(model_path)

# # Loop through each image in the folder
# for img_name in os.listdir(image_folder):
#     if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#         img_path = os.path.join(image_folder, img_name)

#         # Load and resize the image to 640x640
#         image = cv2.imread(img_path)
#         image_resized = cv2.resize(image, (640, 640))

#         # Save the resized image temporarily for inference (YOLO requires path or array)
#         temp_path = "temp_resized_image.jpg"
#         cv2.imwrite(temp_path, image_resized)

#         # Run detection on resized image
#         results = model(temp_path)[0]

#         # Draw boxes on the resized image with confidence threshold > 0.6
#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls_id = int(box.cls[0])
#             conf = float(box.conf[0])

#             if conf < 0.55:  # Skip boxes with confidence less than 0.6
#                 continue

#             # label = f"{custom_names.get(cls_id, str(cls_id))} {conf:.2f}"

#             class_name = custom_names.get(cls_id, str(cls_id))
#             print(f" Detected: {class_name} | Confidence: {conf:.2f}")        

#             label = f"{class_name} {conf:.2f}"
#             cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image_resized, label, (x1, y1 + 100),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Show the detection result
#         cv2.imshow("Detection", image_resized)

#         # Wait for Enter key, or allow 'q' or ESC to quit
#         while True:
#             key = cv2.waitKey(1) & 0xFF
#             if key in [13, 10]:  # Enter key
#                 break
#             elif key in [27, ord('q')]:  # ESC or 'q'
#                 cv2.destroyAllWindows()
#                 exit()

# cv2.destroyAllWindows()

# # Optionally delete the temp file after use
# if os.path.exists("temp_resized_image.jpg"):
#     os.remove("temp_resized_image.jpg")
import cv2
import os
from ultralytics import YOLO

# Set model path and image path
model_path = r"D:\face_vsrun\runs\detect\yolov_train_partimages18\weights\best.pt"
image_path = r"D:\downloads_archieve_14_07_2025\26117\your_image.jpg"  # <-- Change this to your image path

# Custom class names
custom_names = {
    0: "17&16",
    1: "New part",
}

# Load the YOLO model
model = YOLO(model_path)

# Check if the image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load and resize the image to 640x640
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (640, 640))

# Save the resized image temporarily for inference (YOLO requires path or array)
temp_path = "temp_resized_image.jpg"
cv2.imwrite(temp_path, image_resized)

# Run detection on the resized image
results = model(temp_path)[0]

# Draw boxes on the resized image with confidence threshold > 0.55
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])

    if conf < 0.55:
        continue

    class_name = custom_names.get(cls_id, str(cls_id))
    print(f"Detected: {class_name} | Confidence: {conf:.2f}")

    label = f"{class_name} {conf:.2f}"
    cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_resized, label, (x1, y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the detection result
cv2.imshow("Detection", image_resized)

# Wait for Enter key, or allow 'q' or ESC to quit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key in [13, 10]:  # Enter
        break
    elif key in [27, ord('q')]:  # ESC or 'q'
        break

cv2.destroyAllWindows()

# Optionally delete the temp file after use
if os.path.exists(temp_path):
    os.remove(temp_path)
