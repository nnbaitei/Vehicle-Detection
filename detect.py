import cv2
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

def detect_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480)) 

        results = model(frame_resized)

        detections = results.pred[0] 
        for *xyxy, conf, cls in detections:
            if int(cls) == 2:  
                x1, y1, x2, y2 = map(int, xyxy)  
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  
                cv2.putText(frame_resized, f'Car {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Vehicle Detection', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = r"dataset/car.mp4"
detect_vehicles(video_path)

# import torch

# # ตรวจสอบว่า GPU พร้อมใช้งาน
# print("CUDA available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("Current GPU:", torch.cuda.current_device())
