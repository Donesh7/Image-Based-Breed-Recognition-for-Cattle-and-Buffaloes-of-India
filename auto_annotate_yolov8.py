import os
from ultralytics import YOLO

# ✅ Load YOLOv8 model (pretrained on COCO)
model = YOLO("yolov8n.pt")

# ✅ Paths
base_path = r"C:\Users\Dell\DQAI_3\datasets"
dataset_path = os.path.join(base_path, "train")
labels_path = os.path.join(base_path, "labels")
yaml_path = os.path.join(base_path, "data.yaml")

# ✅ Define class mapping (each breed gets unique ID)
breed_classes = {
    "Gir": 0,
    "Sahiwal": 1,
    "Tharparkar": 2,
    "Red_Sindhi": 3,
    "Kankrej": 4,
    "Hariana": 5,
    "Murrah": 6,
    "Jaffarabadi": 7,
    "Surti": 8,
    "Mehsana": 9
}

# ✅ Create labels folder structure
os.makedirs(labels_path, exist_ok=True)
for breed in breed_classes.keys():
    os.makedirs(os.path.join(labels_path, breed), exist_ok=True)

# ✅ Loop through breed folders
for breed, cls_id in breed_classes.items():
    breed_folder = os.path.join(dataset_path, breed)
    breed_label_folder = os.path.join(labels_path, breed)

    if not os.path.isdir(breed_folder):
        print(f"⚠️ Skipping missing folder: {breed_folder}")
        continue

    print(f"🔄 Processing breed: {breed} (class {cls_id})")

    # Loop through images in the breed folder
    for img_file in os.listdir(breed_folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(breed_folder, img_file)

        # Run YOLOv8 inference
        results = model(img_path, verbose=False)

        # Save annotation in YOLO format
        for r in results:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(breed_label_folder, label_file)

            with open(label_path, "w") as f:
                for box in r.boxes.xywhn:  # Normalized [x_center, y_center, w, h]
                    x, y, w, h = box.tolist()
                    f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("✅ All breeds annotated successfully! Labels are ready in YOLO format.")

# ✅ Auto-generate data.yaml
with open(yaml_path, "w") as f:
    f.write("# YOLOv8 dataset configuration\n\n")
    f.write(f"path: {base_path}\n\n")
    f.write("train: train\n")
    f.write("val: train  # (you can later split into val folder)\n")
    f.write("test: \n\n")
    f.write("names:\n")
    for breed, cls_id in breed_classes.items():
        f.write(f"  {cls_id}: {breed}\n")

print(f"✅ data.yaml created at: {yaml_path}")
