import cv2
import numpy as np
from ultralytics import YOLO
import math
import time

# 거리 계산 방식 선택 함수
def calculate_distance(box1, box2, method="edge"):
    """두 개의 바운딩 박스 간의 거리 계산 방식 선택"""
    if method == "edge":
        return edge_distance(box1, box2)
    elif method == "euclidean":
        return euclidean_distance(box1, box2)
    else:
        raise ValueError("Unsupported distance calculation method")

# 엣지 거리 계산 함수
def edge_distance(box1, box2):
    _, _, x1_min, y1_min, x1_max, y1_max = box1
    _, _, x2_min, y2_min, x2_max, y2_max = box2
    dx = max(x1_min - x2_max, x2_min - x1_max, 0)
    dy = max(y1_min - y2_max, y2_min - y1_max, 0)
    return math.sqrt(dx**2 + dy**2)

# 유클리드 거리 계산 함수
def euclidean_distance(box1, box2):
    cx1, cy1, _, _, _, _ = box1
    cx2, cy2, _, _, _, _ = box2
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

# 모델 불러오기
model = YOLO("./50_best.pt")

# 동영상 파일을 입력으로 사용
cap = cv2.VideoCapture("C:/Users/User/Desktop/roboflow data/예빈.mp4")

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

# 충돌 거리 임계값
COLLISION_THRESHOLD = 50  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임 로드 불가 또는 동영상 종료")
        break

    frame = frame.astype(np.uint8)
    results = model.predict(frame, conf=0.3, iou=0.5)
    persons, cars = [], []

    for r in results:
        r_b = r.boxes  
        if r_b.cls is not None:
            for idx in range(len(r_b)):
                x1, y1, x2, y2 = int(r_b.xyxy[idx][0]), int(r_b.xyxy[idx][1]), int(r_b.xyxy[idx][2]), int(r_b.xyxy[idx][3])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  
                if r_b.cls[idx] == 0:
                    persons.append((cx, cy, x1, y1, x2, y2))
                    color, label_text = (0, 255, 0), "Person"
                elif r_b.cls[idx] == 3:
                    cars.append((cx, cy, x1, y1, x2, y2))
                    color, label_text = (0, 255, 255), "Car"
                else:
                    color, label_text = (0, 0, 255), "Other"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1 + 3, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    min_distance, collision_warning = float("inf"), False
    distance_method = "edge"  # "euclidean"으로 변경 가능

    for person in persons:
        closest_car = None
        min_distance = float("inf")

        # 각 person에 대해 가장 가까운 car 찾기
        for car in cars:
            distance = calculate_distance(person, car, distance_method)
            if distance < min_distance:
                min_distance = distance
                closest_car = car  

        # 가장 가까운 car와 충돌 예측
        if closest_car:
            if min_distance < COLLISION_THRESHOLD:
                cv2.putText(frame, 'collision can be occurred', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frame, 'no threat', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 5), 3)
 
    
    cv2.imshow("Collision Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
