from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

# Load YOLO model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov10l-face.pt",   # your model
    confidence_threshold=0.20,
    device="cuda"  # GPU
)

image_path = "crowd.jpg"

# Run sliced prediction
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=180,
    slice_width=180,
    overlap_height_ratio=0.3,
    overlap_width_ratio=0.3,
)

# Convert to OpenCV image
image = cv2.imread(image_path)

count = 0

for obj in result.object_prediction_list:
    bbox = obj.bbox

    x1 = int(bbox.minx)
    y1 = int(bbox.miny)
    x2 = int(bbox.maxx)
    y2 = int(bbox.maxy)

    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

    count += 1

print("People detected:", count)

cv2.imshow("Detection", image)
cv2.waitKey(0)