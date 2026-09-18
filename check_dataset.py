import os
import xml.etree.ElementTree as ET
from collections import Counter

BASE_PATH = "BCCD-Dataset-blood-cells-detection-master/BCCD"

IMAGE_PATH = os.path.join(BASE_PATH, "JPEGImages")
ANNOTATION_PATH = os.path.join(BASE_PATH, "Annotations")

print("Images path exists:", os.path.exists(IMAGE_PATH))
print("Annotations path exists:", os.path.exists(ANNOTATION_PATH))

images = os.listdir(IMAGE_PATH)
annotations = os.listdir(ANNOTATION_PATH)

print("Total images:", len(images))
print("Total annotations:", len(annotations))

classes = Counter()

for file in annotations:
    if file.endswith(".xml"):
        xml_path = os.path.join(ANNOTATION_PATH, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text
            classes[name] += 1

print("\nBlood cell classes:")
for name, count in classes.items():
    print(name, ":", count)