from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("2_models/coin_yolov8_last.pt")

# Coin values by class ID
coin_value = {0: 1, 1: 2, 2: 5, 3: 10}

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on current frame
    results = model(frame, imgsz=640, conf=0.5)[0]

    total_value = 0
    coin_count = 0

    # Process detections
    if results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls)
            coin_count += 1
            total_value += coin_value.get(cls, 0)

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            label = f"Rs {coin_value.get(cls, 0)}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display totals
    cv2.putText(frame, f"Coins: {coin_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Total: Rs {total_value}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show live output
    cv2.imshow("Live Coin Counter", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
