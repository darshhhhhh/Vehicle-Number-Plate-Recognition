# Vehicle Number Plate Recognition using YOLOv8 and EasyOCR

An end-to-end computer vision project for detecting vehicle number plates from images/videos and recognizing the extracted plate text using OCR.

This project uses **YOLOv8** for number plate detection and **EasyOCR** for reading the cropped plate text. It includes model training, single-image prediction, batch prediction, CSV result export, and an interactive **Streamlit dashboard** for visual testing.

---

## Project Overview

Vehicle number plate recognition is a real-world computer vision problem used in traffic monitoring, parking systems, toll collection, security systems, and automated vehicle identification.

This project follows a complete ML workflow:

1. Dataset preparation
2. YOLOv8 model training
3. Number plate detection
4. Plate cropping
5. OCR-based text recognition
6. OCR post-processing
7. Single-image prediction
8. Batch prediction
9. Streamlit dashboard visualization
10. CSV result generation

---

## Features

* Detects vehicle number plates using a custom-trained YOLOv8 model
* Crops detected number plate regions automatically
* Reads plate text using EasyOCR
* Applies OCR post-processing to fix common character mistakes
* Supports single-image prediction
* Supports batch prediction on a folder of images
* Supports image and video upload through Streamlit
* Displays annotated image with bounding box
* Displays extracted plate text and confidence scores
* Displays cropped number plate image
* Exports prediction results to CSV
* Allows downloading annotated output image from the dashboard

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for testing and visualization.

The dashboard allows users to:

* Upload an image or video
* Detect vehicle number plates
* Display the annotated image with bounding boxes
* Extract and display the recognized plate text
* Show cropped number plate regions
* Download the annotated output image

### Dashboard Preview

<p align="center">
  <img src="assets/screenshots/streamlit-dashboard-main.png" alt="Streamlit dashboard showing uploaded and annotated image" width="900">
</p>

<p align="center">
  <img src="assets/screenshots/streamlit-dashboard-results.png" alt="Streamlit dashboard showing extracted plate text and cropped plate" width="900">
</p>

---

## Tech Stack

* Python
* YOLOv8 / Ultralytics
* OpenCV
* EasyOCR
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Pillow

---

## Model Performance

The YOLOv8 nano model was trained for 50 epochs on a custom Indian vehicle number plate dataset.

Final validation performance:

| Metric    | Score |
| --------- | ----: |
| Precision | 0.827 |
| Recall    | 0.853 |
| mAP50     | 0.894 |
| mAP50-95  | 0.386 |

The model performs well for number plate detection. OCR accuracy depends on image quality, plate clarity, lighting conditions, viewing angle, and plate resolution.

---

## Project Structure

```text
Vehicle-Number-Plate-Recognition/
│
├── assets/
│   └── screenshots/
│       ├── streamlit-dashboard-main.png
│       └── streamlit-dashboard-results.png
│
├── data/
│   └── processed/
│       └── data.yaml
│
├── models/
│   └── best.pt
│
├── outputs/
│   ├── detected_images/
│   ├── cropped_plates/
│   ├── results.csv
│   └── batch_results.csv
│
├── src/
│   ├── app.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── predict_image.py
│   ├── predict_batch.py
│   └── ocr.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The dataset used for training contains Indian vehicle number plate images with YOLO-format annotations.

The raw dataset is not included in this repository because of size and ownership reasons.

Dataset source:

```text
Indian Vehicle Number Plate YOLO Annotation Dataset
https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
```

Expected raw dataset structure:

```text
data/raw/
├── vid-1/
├── vid-2/
├── vid-3/
└── classes.txt
```

After running the dataset preparation script, the dataset is converted into YOLOv8 format:

```text
data/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/darshhhhhh/Vehicle-Number-Plate-Recognition.git
cd Vehicle-Number-Plate-Recognition
```

Install required dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Prepare Dataset

Place the downloaded dataset inside:

```text
data/raw/
```

Then run:

```bash
python src/prepare_dataset.py
```

This script prepares the dataset into YOLOv8 format and creates train, validation, and test splits.

---

## Train the YOLOv8 Model

To train the model:

```bash
python src/train.py
```

The trained model will be saved by Ultralytics inside:

```text
runs/detect/plate_yolov8n_50/weights/best.pt
```

Copy the best model to the `models/` folder:

```bash
copy runs\detect\plate_yolov8n_50\weights\best.pt models\best.pt
```

For PowerShell:

```powershell
Copy-Item .\runs\detect\plate_yolov8n_50\weights\best.pt .\models\best.pt
```

---

## Run Streamlit Dashboard

To launch the interactive dashboard:

```bash
streamlit run src/app.py
```

If Streamlit is not recognized, run:

```bash
python -m streamlit run src/app.py
```

The dashboard supports:

* Image upload
* Video upload
* Number plate detection
* OCR text extraction
* Cropped plate visualization
* Annotated image download

---

## Single Image Prediction

Run prediction on one image:

```bash
python src/predict_image.py --image data/processed/images/test/sample_image.jpg
```

Example output:

```text
Plate 1:
Bounding Box: 1039, 887, 1146, 936
Detection Confidence: 0.88
OCR Text: KA01MN4259
OCR Confidence: 0.49
Cropped plate saved at: outputs/cropped_plates/plate_sample_image_1.jpg
```

The detected image will be saved in:

```text
outputs/detected_images/
```

The cropped number plate will be saved in:

```text
outputs/cropped_plates/
```

The result will be saved in:

```text
outputs/results.csv
```

---

## Batch Prediction

To run prediction on all images inside the test folder:

```bash
python src/predict_batch.py
```

By default, this processes images from:

```text
data/processed/images/test/
```

To run batch prediction on a different folder:

```bash
python src/predict_batch.py --source data/processed/images/val
```

To use a lower detection confidence threshold:

```bash
python src/predict_batch.py --source data/processed/images/test --conf 0.15
```

Batch results are saved in:

```text
outputs/batch_results.csv
```

---

## Output Files

After running predictions, the output folder contains:

```text
outputs/
├── detected_images/
│   └── images with number plate bounding boxes
│
├── cropped_plates/
│   └── cropped number plate images
│
├── results.csv
└── batch_results.csv
```

The CSV file contains:

* Image name
* Recognized plate number
* Detection confidence
* OCR confidence
* Bounding box coordinates
* Cropped plate path
* Detected image path

---

## OCR Post-processing

OCR models can confuse visually similar characters, such as:

```text
O ↔ 0
D ↔ 0
B ↔ 8
G ↔ 6
S ↔ 5
I ↔ 1
```

To reduce these mistakes, the OCR output is cleaned using Indian number plate format rules.

Example expected format:

```text
KA01MN4259
```

Pattern:

```text
LLNNLLNNNN
```

Where:

```text
L = Letter
N = Number
```

This helps correct common OCR mistakes such as:

```text
KAO1MN4259 → KA01MN4259
```

---

## Example Result

Input:

```text
Vehicle image containing a visible number plate
```

Output:

```text
Detected Plate: KA01MN4259
Detection Confidence: 0.88
OCR Confidence: 0.49
```

Generated files:

```text
outputs/detected_images/detected_sample_image.jpg
outputs/cropped_plates/plate_sample_image_1.jpg
outputs/batch_results.csv
```

---

## Future Improvements

* Improve OCR accuracy using a character-level recognition model
* Add better video output preview support
* Add real-time webcam detection
* Train on a larger and more diverse dataset
* Add Indian state-code validation
* Add deployment using Streamlit Cloud or Hugging Face Spaces
* Add Docker support for easier setup

---

## Skills Demonstrated

This project demonstrates:

* Computer vision
* Object detection
* YOLOv8 training
* OCR integration
* Image preprocessing
* Model inference
* Batch processing
* CSV report generation
* Streamlit dashboard development
* Clean Python project structure

---

## Author

**Darshil Vaja**
