import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from ocr import PlateOCR


MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_ocr_reader():
    return PlateOCR()


def detect_plate_from_image(image_bgr, model, ocr_reader, conf_threshold):
    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        save=False,
        verbose=False
    )

    annotated_image = image_bgr.copy()
    detections = []
    cropped_plates = []

    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            continue

        for idx, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detection_confidence = float(box.conf[0])

            h, w, _ = image_bgr.shape

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cropped_plate = image_bgr[y1:y2, x1:x2]

            if cropped_plate.size == 0:
                continue

            plate_text, ocr_confidence = ocr_reader.read_plate(cropped_plate)

            if plate_text == "":
                plate_text = "NOT_READ"

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{plate_text} | {detection_confidence:.2f}"

            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            detections.append({
                "Plate Number": plate_text,
                "Detection Confidence": round(detection_confidence, 4),
                "OCR Confidence": round(ocr_confidence, 4),
                "Bounding Box": f"{x1}, {y1}, {x2}, {y2}"
            })

            cropped_plates.append(cropped_plate)

    return annotated_image, detections, cropped_plates


def convert_bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def get_image_download_bytes(image_bgr):
    success, encoded_image = cv2.imencode(".jpg", image_bgr)

    if not success:
        return None

    return encoded_image.tobytes()


def process_video(uploaded_video, model, ocr_reader, conf_threshold, frame_skip):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    video_capture = cv2.VideoCapture(temp_input.name)

    if not video_capture.isOpened():
        return None, [], "Could not open video file."

    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = 25

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        temp_output.name,
        fourcc,
        fps,
        (width, height)
    )

    all_detections = []
    frame_number = 0

    progress_bar = st.progress(0)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame_number += 1

        if frame_number % frame_skip == 0:
            annotated_frame, detections, _ = detect_plate_from_image(
                frame,
                model,
                ocr_reader,
                conf_threshold
            )

            for detection in detections:
                detection["Frame"] = frame_number
                all_detections.append(detection)

            video_writer.write(annotated_frame)
        else:
            video_writer.write(frame)

        if total_frames > 0:
            progress_bar.progress(min(frame_number / total_frames, 1.0))

    video_capture.release()
    video_writer.release()

    return temp_output.name, all_detections, None


def main():
    st.set_page_config(
        page_title="Vehicle Number Plate Recognition",
        page_icon="🚗",
        layout="wide"
    )

    st.title("Vehicle Number Plate Recognition")
    st.write(
        "Upload an image or video, detect vehicle number plates, extract plate text, "
        "and download the annotated output."
    )

    if not Path(MODEL_PATH).exists():
        st.error("Model file not found. Please make sure models/best.pt exists.")
        return

    model = load_yolo_model()
    ocr_reader = load_ocr_reader()

    st.sidebar.header("Settings")

    conf_threshold = st.sidebar.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    frame_skip = st.sidebar.slider(
        "Video frame processing interval",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        help="For videos, OCR will run every N frames. Lower values are slower but more detailed."
    )

    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is None:
        st.info("Upload an image or video to start detection.")
        return

    file_extension = Path(uploaded_file.name).suffix.lower()

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]

    if file_extension in image_extensions:
        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting number plate and extracting text..."):
            annotated_image, detections, cropped_plates = detect_plate_from_image(
                image_bgr,
                model,
                ocr_reader,
                conf_threshold
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.subheader("Annotated Image")
            st.image(convert_bgr_to_rgb(annotated_image), use_container_width=True)

        if detections:
            st.subheader("Extracted Plate Text")
            detections_df = pd.DataFrame(detections)
            st.dataframe(detections_df, use_container_width=True)

            st.subheader("Cropped Plate")
            crop_columns = st.columns(min(len(cropped_plates), 4))

            for idx, cropped_plate in enumerate(cropped_plates):
                with crop_columns[idx % len(crop_columns)]:
                    st.image(
                        convert_bgr_to_rgb(cropped_plate),
                        caption=f"Plate {idx + 1}",
                        use_container_width=True
                    )
        else:
            st.warning("No number plate detected. Try lowering the confidence threshold.")

        download_bytes = get_image_download_bytes(annotated_image)

        if download_bytes:
            st.download_button(
                label="Download Annotated Image",
                data=download_bytes,
                file_name=f"annotated_{uploaded_file.name}",
                mime="image/jpeg"
            )

    elif file_extension in video_extensions:
        st.video(uploaded_file)

        if st.button("Process Video"):
            with st.spinner("Processing video. This may take some time..."):
                output_video_path, detections, error = process_video(
                    uploaded_file,
                    model,
                    ocr_reader,
                    conf_threshold,
                    frame_skip
                )

            if error:
                st.error(error)
                return

            st.success("Video processing completed.")

            st.subheader("Extracted Plate Text from Video")

            if detections:
                detections_df = pd.DataFrame(detections)
                detections_df = detections_df[
                    ["Frame", "Plate Number", "Detection Confidence", "OCR Confidence", "Bounding Box"]
                ]

                st.dataframe(detections_df, use_container_width=True)

                unique_plates = detections_df["Plate Number"].value_counts().reset_index()
                unique_plates.columns = ["Plate Number", "Count"]

                st.subheader("Detected Plate Frequency")
                st.dataframe(unique_plates, use_container_width=True)
            else:
                st.warning("No number plate detected in the processed video frames.")

            with open(output_video_path, "rb") as file:
                video_bytes = file.read()

            st.download_button(
                label="Download Annotated Video",
                data=video_bytes,
                file_name=f"annotated_{uploaded_file.name}",
                mime="video/mp4"
            )

    else:
        st.error("Unsupported file type.")


if __name__ == "__main__":
    main()