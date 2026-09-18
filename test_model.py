from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="BCCD-Dataset-blood-cells-detection-master/example.jpg",
    conf=0.25,
    save=True
)

for result in results:
    counts = {
        "RBC": 0,
        "WBC": 0,
        "Platelets": 0
    }

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in counts:
            counts[class_name] += 1

    print("\nDetected Blood Cells:")
    print("RBC:", counts["RBC"])
    print("WBC:", counts["WBC"])
    print("Platelets:", counts["Platelets"])

print("\nDetection completed!")