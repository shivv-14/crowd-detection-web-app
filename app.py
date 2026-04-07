from flask import Flask, render_template, request
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO Face model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov10l-face.pt",
    confidence_threshold=0.18,
    device="cuda"
)

@app.route("/", methods=["GET", "POST"])
def index():

    count = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Run sliced detection
            result = get_sliced_prediction(
                filepath,
                detection_model,
                slice_height=192,
                slice_width=192,
                overlap_height_ratio=0.30,
                overlap_width_ratio=0.30,
            )

            image = cv2.imread(filepath)

            count = 0

            for obj in result.object_prediction_list:

                bbox = obj.bbox
                score = obj.score.value  # confidence score

                x1 = int(bbox.minx)
                y1 = int(bbox.miny)
                x2 = int(bbox.maxx)
                y2 = int(bbox.maxy)

                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

                # Convert score to percentage
                confidence = int(score * 100)

                label = f"{confidence}%"

                # Draw text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

                count += 1

            output_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
            cv2.imwrite(output_path, image)

            image_path = output_path

    return render_template(
        "index.html",
        count=count,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)