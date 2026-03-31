from ultralytics import YOLO
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "text_detection", "yolo11_text", "weights", "best.pt")

TEST_IMAGE = os.path.join(BASE_DIR, "data", "raw", "images", "img001.jpg")

model = YOLO(MODEL_PATH)

results = model(TEST_IMAGE, conf=0.3)

img = cv2.imread(TEST_IMAGE)

for box in results[0].boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

save_path = os.path.join(BASE_DIR, "test_result.jpg")
cv2.imwrite(save_path, img)

print("Đã lưu kết quả tại:", save_path)