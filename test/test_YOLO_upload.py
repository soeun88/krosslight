import cv2
import numpy as np
from ultralytics import YOLO 

# 모델 불러오기
model = YOLO("C:/Users/User/Desktop/roboflow data/new_best_100.pt")

# 동영상 파일 입력
video_path = "C:/Users/User/Desktop/roboflow data/효인.mp4"
cap = cv2.VideoCapture(video_path)

# 동영상이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print('동영상 파일을 열 수 없습니다.')
    exit()

# 원본 비디오 속성 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 프레임 속도
frame_skip = 1# 프레임 속도 재생

# 첫 번째 프레임을 가져와 크기 확인
ret, frame = cap.read()
if not ret:
    print('비디오를 읽을 수 없습니다.')
    cap.release()
    exit()

height, width, _ = frame.shape  # 원본 프레임 크기
print(f"📏 원본 프레임 크기: {width} x {height}")

# 프레임 회전 (세로 영상일 경우)
frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  
rotated_height, rotated_width, _ = frame_rotated.shape  # 회전 후 크기 확인

# 출력 동영상 설정 (회전 여부에 따라 크기 변경)
output_path = "output_video.mp4"  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps*frame_skip, (rotated_width, rotated_height))

# 프레임마다 YOLO 객체 감지 수행
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('동영상이 끝났습니다.')
        break

    # 프레임 스킵 적용 (프레임 건너뛰기)
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    frame_count += 1
    frame = frame.astype(np.uint8)

    # 프레임 회전 (세로 영상일 경우)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  

    # YOLO 모델 예측
    results = model(frame)

    # 감지된 객체 정보 처리
    for r in results:
        r_b = r.boxes  # 감지된 바운딩 박스 정보
        if r_b is not None:
            for idx in range(len(r_b)):
                x1, y1, x2, y2 = int(r_b.xyxy[idx][0]), int(r_b.xyxy[idx][1]), int(r_b.xyxy[idx][2]), int(r_b.xyxy[idx][3])
                conf = r_b.conf[idx] * 100  # 신뢰도 변환
                
                # 객체 클래스 확인
                if r_b.cls[idx] == 0:  
                    color = (0, 255, 0)  # 사람 - 초록색
                    label_text = f'People'
                elif r_b.cls[idx] == 1:  
                    color = (0, 0, 255)  # 오토바이 - 빨간색
                    label_text = f'Bike'
                elif r_b.cls[idx] == 2:  
                    color = (0, 0, 255)  # 버스 - 빨간색
                    label_text = f'Bus'
                elif r_b.cls[idx] == 3:  
                    color = (0, 0, 255)  # 자동차 - 빨간색
                    label_text = f'Car'
                else:
                    color = (0, 0, 255)  
                    label_text = f'Other'
                    
                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 텍스트 추가
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2)

    # 결과 영상 저장
    out.write(frame)

    # 화면에 출력 (실시간 확인용)
    cv2.imshow('YOLO Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
