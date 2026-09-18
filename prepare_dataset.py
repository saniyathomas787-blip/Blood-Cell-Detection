import os
import shutil
import random
import xml.etree.ElementTree as ET

SOURCE = "BCCD-Dataset-blood-cells-detection-master/BCCD"
IMAGE_DIR = os.path.join(SOURCE, "JPEGImages")
ANNOTATION_DIR = os.path.join(SOURCE, "Annotations")

OUTPUT = "blood_dataset"

classes = {
    "RBC": 0,
    "WBC": 1,
    "Platelets": 2
}

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "labels", split), exist_ok=True)

xml_files = [
    f for f in os.listdir(ANNOTATION_DIR)
    if f.endswith(".xml")
]

random.shuffle(xml_files)

total = len(xml_files)

train_end = int(total * 0.70)
val_end = int(total * 0.85)

splits = {
    "train": xml_files[:train_end],
    "val": xml_files[train_end:val_end],
    "test": xml_files[val_end:]
}

for split, files in splits.items():

    for xml_file in files:

        xml_path = os.path.join(ANNOTATION_DIR, xml_file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        image_source = os.path.join(IMAGE_DIR, filename)
        image_destination = os.path.join(
            OUTPUT, "images", split, filename
        )

        shutil.copy2(image_source, image_destination)

        label_name = os.path.splitext(filename)[0] + ".txt"

        label_path = os.path.join(
            OUTPUT, "labels", split, label_name
        )

        with open(label_path, "w") as label_file:

            for obj in root.findall("object"):

                class_name = obj.find("name").text

                if class_name not in classes:
                    continue

                class_id = classes[class_name]

                box = obj.find("bndbox")

                xmin = float(box.find("xmin").text)
                ymin = float(box.find("ymin").text)
                xmax = float(box.find("xmax").text)
                ymax = float(box.find("ymax").text)

                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                label_file.write(
                    f"{class_id} {x_center} {y_center} "
                    f"{box_width} {box_height}\n"
                )

print("Dataset preparation completed!")
print("Train images:", len(splits["train"]))
print("Validation images:", len(splits["val"]))
print("Test images:", len(splits["test"]))