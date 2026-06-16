from pathlib import Path
import shutil
import random
import argparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_dirs(output_dir):
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_old_processed_data(output_dir):
    """
    Clears old images and labels so we do not mix old incorrect data
    with the newly prepared dataset.
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"

    if images_dir.exists():
        shutil.rmtree(images_dir)

    if labels_dir.exists():
        shutil.rmtree(labels_dir)


def find_vid_folders(raw_dir):
    """
    Finds vid-1, vid-2, vid-3 folders inside data/raw.
    This also works if the dataset is nested inside another folder.
    """
    raw_dir = Path(raw_dir)

    expected_names = {"vid-1", "vid-2", "vid-3"}
    found_folders = []

    for folder in raw_dir.rglob("*"):
        if folder.is_dir() and folder.name.lower() in expected_names:
            found_folders.append(folder)

    return found_folders


def find_label_file(image_path):
    """
    YOLO label file usually has the same name as the image.

    Example:
    image: frame_001.jpg
    label: frame_001.txt
    """
    label_path = image_path.with_suffix(".txt")

    if label_path.exists():
        return label_path

    return None


def copy_files(pairs, output_dir, split):
    for image_path, label_path in pairs:
        # Add folder prefix to avoid duplicate filenames from vid-1, vid-2, vid-3
        prefix = image_path.parent.name

        new_image_name = f"{prefix}_{image_path.name}"
        new_label_name = f"{prefix}_{image_path.stem}.txt"

        image_dest = output_dir / "images" / split / new_image_name
        label_dest = output_dir / "labels" / split / new_label_name

        shutil.copy2(image_path, image_dest)
        shutil.copy2(label_path, label_dest)


def create_yaml(output_dir):
    yaml_content = """path: data/processed

train: images/train
val: images/val
test: images/test

names:
  0: number_plate
"""

    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)


def main(raw_dir, output_dir):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    print(f"Searching dataset inside: {raw_dir.resolve()}")

    if not raw_dir.exists():
        print("ERROR: data/raw folder does not exist.")
        print("Please create data/raw and put the dataset inside it.")
        return

    vid_folders = find_vid_folders(raw_dir)

    if len(vid_folders) == 0:
        print("ERROR: Could not find vid-1, vid-2, or vid-3 folders.")
        print("Your data/raw folder currently contains:")

        for item in raw_dir.iterdir():
            print("-", item.name)

        print("\nMove vid-1, vid-2, vid-3, and classes.txt inside data/raw.")
        return

    print("\nFound video folders:")
    for folder in vid_folders:
        print("-", folder)

    image_files = []

    for folder in vid_folders:
        for file in folder.rglob("*"):
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(file)

    print(f"\nTotal images found: {len(image_files)}")

    pairs = []

    for image_path in image_files:
        label_path = find_label_file(image_path)

        if label_path is None:
            print(f"Label missing for image: {image_path}")
            continue

        pairs.append((image_path, label_path))

    print(f"Valid image-label pairs found: {len(pairs)}")

    if len(pairs) == 0:
        print("\nERROR: No valid image-label pairs found.")
        print("Each image should have a matching .txt file with the same name.")
        print("Example:")
        print("image: frame_001.jpg")
        print("label: frame_001.txt")
        return

    random.seed(42)
    random.shuffle(pairs)

    train_end = int(0.8 * len(pairs))
    val_end = int(0.9 * len(pairs))

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    clear_old_processed_data(output_dir)
    create_dirs(output_dir)

    copy_files(train_pairs, output_dir, "train")
    copy_files(val_pairs, output_dir, "val")
    copy_files(test_pairs, output_dir, "test")

    create_yaml(output_dir)

    print("\nDataset prepared successfully.")
    print(f"Train: {len(train_pairs)}")
    print(f"Val: {len(val_pairs)}")
    print(f"Test: {len(test_pairs)}")
    print(f"YAML created at: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    main(args.raw_dir, args.output_dir)