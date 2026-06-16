from pathlib import Path
import argparse
import csv
import cv2
from ultralytics import YOLO
from ocr import PlateOCR


def save_result_to_csv(csv_path, image_name, plate_number, confidence, crop_path):
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "image_name",
                "plate_number",
                "ocr_confidence",
                "crop_path"
            ])

        writer.writerow([
            image_name,
            plate_number,
            round(confidence, 4),
            crop_path
        ])


def predict_image(image_path, model_path="models/best.pt", conf=0.25):
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    detected_output_dir = Path("outputs/detected_images")
    cropped_output_dir = Path("outputs/cropped_plates")
    results_csv_path = Path("outputs/results.csv")

    detected_output_dir.mkdir(parents=True, exist_ok=True)
    cropped_output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    ocr_reader = PlateOCR()

    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    plate_count = 0

    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            print("No number plate detected.")
            return

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detection_confidence = float(box.conf[0])

            h, w, _ = image.shape

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cropped_plate = image[y1:y2, x1:x2]

            if cropped_plate.size == 0:
                print("Empty crop found. Skipping.")
                continue

            plate_count += 1

            cropped_plate_path = cropped_output_dir / f"plate_{image_path.stem}_{plate_count}.jpg"
            cv2.imwrite(str(cropped_plate_path), cropped_plate)

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

            label = f"{plate_text} | Det: {detection_confidence:.2f}"

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            save_result_to_csv(
                results_csv_path,
                image_path.name,
                plate_text,
                ocr_confidence,
                str(cropped_plate_path)
            )

            print(f"\nPlate {plate_count}:")
            print(f"Bounding Box: {x1}, {y1}, {x2}, {y2}")
            print(f"Detection Confidence: {detection_confidence:.2f}")
            print(f"OCR Text: {plate_text}")
            print(f"OCR Confidence: {ocr_confidence:.2f}")
            print(f"Cropped plate saved at: {cropped_plate_path}")

    detected_output_path = detected_output_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(detected_output_path), image)

    print(f"\nDetected image saved at: {detected_output_path}")
    print(f"Results saved at: {results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLO model")
    parser.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")

    args = parser.parse_args()

    predict_image(args.image, args.model, args.conf)