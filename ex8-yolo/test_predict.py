import cv2
from ultralytics import YOLO

# Tải mô hình YOLOv5 đã được huấn luyện sẵn
model = YOLO('yolov5s.pt')  # Có thể thay bằng các mô hình khác như 'yolov5m.pt', 'yolov5l.pt'

# Mở camera (0 là camera mặc định, nếu có nhiều camera có thể thử với 1, 2, v.v.)
cap = cv2.VideoCapture(0)

while True:
    # Đọc một frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera.")
        break

    # Dự đoán đối tượng trên frame
    results = model(frame)

    # Lấy các kết quả bounding boxes, labels và confidences
    boxes = results[0].boxes.xyxy  # Lấy tọa độ bounding boxes (xyxy format)
    labels = results[0].names  # Lấy tên của các đối tượng được nhận diện
    confidences = results[0].boxes.conf  # Lấy độ tin cậy của các bounding boxes

    # Vẽ các bounding boxes lên frame
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])  # Chuyển đổi tọa độ thành integer
        label = labels[int(results[0].boxes.cls[i])]  # Lấy tên đối tượng
        confidence = confidences[i]  # Lấy độ tin cậy của bounding box

        # Vẽ hình chữ nhật và tên đối tượng lên frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị frame với bounding boxes
    cv2.imshow('YOLOv5 Prediction', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
