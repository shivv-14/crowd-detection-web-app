# Crowd Detection using YOLOv10 🚀

## 📌 Overview

This project is a **Deep Learning-based Crowd Detection Web Application** built using **YOLOv10 (Large model)** and **SAHI (Sliced Inference)**.

The system detects and counts people (faces) in an image by performing high-accuracy object detection, even in dense crowd scenarios.

---

## 🧠 Key Features

* 📸 Upload image via web interface
* 👥 Detect multiple faces in crowded scenes
* 🔍 Uses **SAHI slicing** for better small-object detection
* 📊 Displays bounding boxes with confidence scores
* 🔢 Shows total crowd count
* ⚡ GPU acceleration using CUDA

---

## 🏗️ Tech Stack

* **Programming Language:** Python
* **Framework:** Flask
* **Deep Learning:** YOLOv10 (Ultralytics)
* **Inference Optimization:** SAHI (Sliced Prediction)
* **Computer Vision:** OpenCV
* **Frontend:** HTML (Jinja Templates)

---

## 📂 Project Structure

```
crowd_web_app/
│
├── app.py                  # Main Flask app
├── test.py                 # Standalone detection script
├── slice_test.py           # GPU / environment check
│
├── templates/
│   └── index.html          # UI page
│
├── static/
│   └── uploads/            # Uploaded + result images
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ How It Works

1. User uploads an image
2. Image is saved in `static/uploads`
3. SAHI splits image into smaller slices
4. YOLOv10 detects faces in each slice
5. Results are merged
6. Bounding boxes + confidence scores are drawn
7. Total count is displayed

---

## 🧪 Core Components

### 🔹 YOLOv10 Model

* Model used: `yolov10l-face.pt`
* Confidence threshold: ~0.18–0.20
* Runs on GPU using CUDA

---

### 🔹 SAHI (Sliced Prediction)

Used to improve detection in dense crowds:

```python
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=192,
    slice_width=192,
    overlap_height_ratio=0.30,
    overlap_width_ratio=0.30,
)
```

---

### 🔹 Flask App

* Handles image upload
* Runs detection
* Displays results on UI

---

## ▶️ How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/crowd-detection-yolov10.git
cd crowd-detection-yolov10
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 Output

* Bounding boxes around detected faces
* Confidence score displayed on each detection
* Total number of people detected

---

## ⚠️ Notes

* Model file (`.pt`) is not uploaded due to GitHub size limits
* Add your model manually or provide a download link

---

## 🚀 Future Improvements

* 🎥 Real-time video crowd detection
* 📈 Crowd density estimation
* 📊 Analytics dashboard
* 🌐 Cloud deployment (AWS / Render / Railway)

---

## 👨‍💻 Author

**Brungi Shiva Ganesh**

---

## 🏷️ Domain

**Deep Learning | Computer Vision | Object Detection**
