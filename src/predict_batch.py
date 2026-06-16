from pathlib import Path
import argparse
import csv
import cv2
from ultralytics import YOLO
from ocr import PlateOCR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_output_dirs():
    detected_output_dir = Path("outputs/detected_images")
    cropped_output_dir = Path("outputs/cropped_plates")

    detected_output_dir.mkdir(parents=True, exist_ok=True)
    cropped_output_dir.mkdir(parents=True, exist_ok=True)

    return detected_output_dir, cropped_output_dir


def write_csv_header(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_name",
            "plate_number",
            "detection_confidence",
            "ocr_confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "crop_path",
            "detected_image_path"
        ])


def save_result_to_csv(
    csv_path,
    image_name,
    plate_number,
    detection_confidence,
    ocr_confidence,
    bbox,
    crop_path,
    detected_image_path
):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        x1, y1, x2, y2 = bbox

        writer.writerow([
            image_name,
            plate_number,
            round(detection_confidence, 4),
            round(ocr_confidence, 4),
            x1,
            y1,
            x2,
            y2,
            crop_path,
            detected_image_path
        ])


def get_images(source_path):
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    if source_path.is_file():
        if source_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [source_path]
        raise ValueError(f"Unsupported image format: {source_path.suffix}")

    image_paths = []

    for file in source_path.rglob("*"):
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(file)

    return sorted(image_paths)


def process_single_image(
    image_path,
    model,
    ocr_reader,
    detected_output_dir,
    cropped_output_dir,
    csv_path,
    conf
):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        verbose=False
    )

    plate_count = 0
    detected_any_plate = False

    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            print(f"No plate detected: {image_path.name}")
            continue

        for box in boxes:
            detected_any_plate = True
            plate_count += 1

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detection_confidence = float(box.conf[0])

            h, w, _ = image.shape

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cropped_plate = image[y1:y2, x1:x2]

            if cropped_plate.size == 0:
                print(f"Empty crop skipped: {image_path.name}")
                continue

            crop_filename = f"plate_{image_path.stem}_{plate_count}.jpg"
            crop_path = cropped_output_dir / crop_filename
            cv2.imwrite(str(crop_path), cropped_plate)

            plate_text, ocr_confidence = ocr_reader.read_plate(cropped_plate)

            if plate_text == "":
                plate_text = "NOT_READ"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{plate_text} | {detection_confidence:.2f}"

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            detected_image_path = detected_output_dir / f"detected_{image_path.name}"

            save_result_to_csv(
                csv_path=csv_path,
                image_name=image_path.name,
                plate_number=plate_text,
                detection_confidence=detection_confidence,
                ocr_confidence=ocr_confidence,
                bbox=(x1, y1, x2, y2),
                crop_path=str(crop_path),
                detected_image_path=str(detected_image_path)
            )

            print(
                f"{image_path.name} → {plate_text} "
                f"| Detection: {detection_confidence:.2f} "
                f"| OCR: {ocr_confidence:.2f}"
            )

    detected_image_path = detected_output_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(detected_image_path), image)

    if not detected_any_plate:
        save_result_to_csv(
            csv_path=csv_path,
            image_name=image_path.name,
            plate_number="NO_DETECTION",
            detection_confidence=0.0,
            ocr_confidence=0.0,
            bbox=(0, 0, 0, 0),
            crop_path="",
            detected_image_path=str(detected_image_path)
        )


def predict_batch(source, model_path="models/best.pt", conf=0.25):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    detected_output_dir, cropped_output_dir = create_output_dirs()

    csv_path = Path("outputs/batch_results.csv")
    write_csv_header(csv_path)

    image_paths = get_images(source)

    if len(image_paths) == 0:
        print("No images found.")
        return

    print(f"Total images found: {len(image_paths)}")
    print("Loading YOLO model...")
    model = YOLO(str(model_path))

    print("Loading OCR model...")
    ocr_reader = PlateOCR()

    print("\nStarting batch prediction...\n")

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Processing: {image_path.name}")

        process_single_image(
            image_path=image_path,
            model=model,
            ocr_reader=ocr_reader,
            detected_output_dir=detected_output_dir,
            cropped_output_dir=cropped_output_dir,
            csv_path=csv_path,
            conf=conf
        )

    print("\nBatch prediction completed.")
    print(f"Detected images saved in: {detected_output_dir}")
    print(f"Cropped plates saved in: {cropped_output_dir}")
    print(f"CSV results saved at: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/processed/images/test",
        help="Path to image folder or single image"
    )
    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to trained YOLO model"
    )
    parser.add_argument(
        "--conf",
        default=0.25,
        type=float,
        help="Detection confidence threshold"
    )

    args = parser.parse_args()

    predict_batch(
        source=args.source,
        model_path=args.model,
        conf=args.conf
    )