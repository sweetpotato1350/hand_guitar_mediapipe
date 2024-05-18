#3. 영역 그리기 (2)박스+기타줄
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
    line_color = (10, 10, 10, 255)
    line_color1 = (255, 255, 255)
    for hor_line_gen in range(6):
        cv2.line(image, (0, 100 + 75 * hor_line_gen), (1500, 100 + 75 * hor_line_gen), line_color,
                 3)  # Line 생성(가로줄)
        cv2.rectangle(image, (970, 80 + 75 * hor_line_gen), (1190, 120 + 75 * hor_line_gen), (255, 0, 0),
                      6)  # box 1
        cv2.rectangle(image, (730, 80 + 75 * hor_line_gen), (950, 120 + 75 * hor_line_gen), (0, 255, 0), 6)  # box 2
        cv2.rectangle(image, (490, 80 + 75 * hor_line_gen), (710, 120 + 75 * hor_line_gen), (0, 0, 255), 6)  # box 3
        cv2.rectangle(image, (250, 80 + 75 * hor_line_gen), (470, 120 + 75 * hor_line_gen), (0, 255, 255),
                      6)  # box 4
        cv2.rectangle(image, (230, 80 + 75 * hor_line_gen), (9, 120 + 75 * hor_line_gen), (255, 0, 255), 6)  # box 5
        cv2.line(image, (960, 100), (960, 475), line_color, 3)  # vertical Line 1
        cv2.line(image, (720, 100), (720, 475), line_color, 3)  # vertical Line 2
        cv2.line(image, (480, 100), (480, 475), line_color, 3)
        cv2.line(image, (240, 100), (240, 475), line_color, 3)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()