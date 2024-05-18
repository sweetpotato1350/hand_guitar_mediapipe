#1. 캠을 켜보자
import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()