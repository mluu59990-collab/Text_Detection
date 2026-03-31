import os
import cv2
import csv
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_ROOT = os.path.join(BASE_DIR, "data", "raw", "img")
XML_PATH = os.path.join(BASE_DIR, "data", "raw", "annotation_xml", "words.xml")

CRNN_ROOT = os.path.join(BASE_DIR, "data", "crnn_recognition")
CROP_DIR = os.path.join(CRNN_ROOT, "crops")
CSV_PATH = os.path.join(CRNN_ROOT, "labels.csv")

os.makedirs(CROP_DIR, exist_ok=True)

tree = ET.parse(XML_PATH)
root = tree.getroot()

rows = []
crop_count = 0

for image_node in root.findall("image"):
    image_name_node = image_node.find("imageName")
    tagged_rectangles_node = image_node.find("taggedRectangles")

    if image_name_node is None or tagged_rectangles_node is None:
        continue

    rel_image_path = image_name_node.text.strip()
    src_image_path = os.path.join(IMG_ROOT, rel_image_path)

    if not os.path.exists(src_image_path):
        continue

    img = cv2.imread(src_image_path)
    if img is None:
        continue

    image_stem = rel_image_path.replace("/", "__").replace("\\", "__")
    rects = tagged_rectangles_node.findall("taggedRectangle")

    for i, rect in enumerate(rects):
        tag_node = rect.find("tag")
        if tag_node is None:
            continue

        text_label = tag_node.text.strip()
        if text_label == "":
            continue

        x = int(float(rect.attrib["x"]))
        y = int(float(rect.attrib["y"]))
        w = int(float(rect.attrib["width"]))
        h = int(float(rect.attrib["height"]))

        if w <= 0 or h <= 0:
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        crop_name = f"{os.path.splitext(image_stem)[0]}_{i}.jpg"
        crop_path = os.path.join(CROP_DIR, crop_name)

        cv2.imwrite(crop_path, crop)

        rows.append([crop_path, text_label])
        crop_count += 1

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(rows)

print("Đã tạo xong CRNN data")
print("Số crop:", crop_count)
print("CSV:", CSV_PATH)