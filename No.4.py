# 4 닿으면 인식되게 만들기
import time
import cv2  # opencv라이브러리 import
import mediapipe as mp  # MediaPipe패키지 import하고 mp로 사용
import numpy as np
import pygame
from math import hypot

from comtypes import CLSCTX_ALL

pygame.init()

size = [200, 200]
screen = pygame.display.set_mode(size)
pygame.display.set_caption("practice")

# mediapipe 패키지에서 사용할 기능들
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  # 손 인식을 위한 객체

box_dect = [[0] * 6 for _ in range(5)]
circle_dect = 0
# To access speaker through the library pycaw

cap = cv2.VideoCapture(0)  # 비디오 캡쳐 객체를 생성/안의 숫자는 장치 인덱스(어떤 카메라를 사용할 것인가)
# 1개만 부착되어 있으면 0, 2개 이상이면 첫 웹캠은 0, 두번째 웹캠은 1으로 지정합니다
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():  # 특정 키를 누를때까지 무한 반복
        success, image = cap.read()  # 비디오의 한 프레임씩 읽습니다.
        image = cv2.resize(image, (1200, 800))
        # 제대로 프레임을 읽으면 ret값이 True, 실패하면 False가 나타납니다. fram에 읽은 프레임이 나옵니다
        if not success:
            print("카메라를 찾을 수 없습니다.")  ## 카메라가 열렸는지 확인# 열리지 않았으면 문자열 출력
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            continue

            # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False  # 이미지 다시쓰기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 형식 전환
        results = hands.process(image)  # AI 모델을 가동시켜 손모양을 인식

        image.flags.writeable = True  # 이미지를 RGB로 나타냄
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지를 RGB로 나타냄
        circle_dect = 0
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

        lmList = []  # empty list

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):  # adding counter and returning it
                    # Get finger joint points
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])  # adding to the empty list 'lmList'
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger1 = hand_landmarks.landmark[4]
                finger1_x_pos = finger1.x * 1200
                finger1_y_pos = finger1.y * 800
                finger2 = hand_landmarks.landmark[8]
                finger2_x_pos = finger2.x * 1200
                finger2_y_pos = finger2.y * 800
                finger3 = hand_landmarks.landmark[12]
                finger3_x_pos = finger3.x * 1200
                finger3_y_pos = finger3.y * 800
                finger4 = hand_landmarks.landmark[16]
                finger4_x_pos = finger4.x * 1200
                finger4_y_pos = finger4.y * 800
                finger5 = hand_landmarks.landmark[20]
                finger5_x_pos = finger5.x * 1200
                finger5_y_pos = finger5.y * 800

                # if (finger2_x_pos >= 1050 and finger2_x_pos <= 1100):
                #   if (finger2_y_pos >= 650 and finger2_y_pos <= 700 ):
                #      circle_dect = 1

                for i in range(6):
                    # 검지
                    if (finger2_x_pos >= 970 and finger2_x_pos <= 1190):
                        if (finger2_y_pos >= (80 + (i * 75)) and finger2_y_pos <= 120 + (75 * i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (970, 80 + (i * 75)), (1190, 120 + (75 * i)), line_color1, -1)
                    if (finger2_x_pos >= 730 and finger2_x_pos <= 950):
                        if (finger2_y_pos >= (80 + (i * 75)) and finger2_y_pos <= 120 + (75 * i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (730, 80 + (i * 75)), (950, 120 + (75 * i)), line_color1, -1)
                    if (finger2_x_pos >= 490 and finger2_x_pos <= 710):
                        if (finger2_y_pos >= (80 + (i * 75)) and finger2_y_pos <= 120 + (75 * i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (490, 80 + (i * 75)), (710, 120 + (75 * i)), line_color1, -1)
                    if (finger2_x_pos >= 250 and finger2_x_pos <= 470):
                        if (finger2_y_pos >= (80 + (i * 75)) and finger2_y_pos <= 120 + (75 * i)):
                            box_dect[3][i] = 1
                            cv2.rectangle(image, (250, 80 + (i * 75)), (470, 120 + (75 * i)), line_color1, -1)
                    if (finger2_x_pos >= 9 and finger2_x_pos <= 230):
                        if (finger2_y_pos >= (80 + (i * 75)) and finger2_y_pos <= 120 + (75 * i)):
                            box_dect[4][i] = 1
                            cv2.rectangle(image, (9, 80 + (i * 75)), (230, 120 + (75 * i)), line_color1, -1)
                    # 엄지
                    if (finger1_x_pos >= 970 and finger1_x_pos <= 1190):
                        if (finger1_y_pos >= (80 + (i * 75)) and finger1_y_pos <= 120 + (75 * i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (970, 80 + (i * 75)), (1190, 120 + (75 * i)), line_color1, -1)
                    if (finger1_x_pos >= 730 and finger1_x_pos <= 950):
                        if (finger1_y_pos >= (80 + (i * 75)) and finger1_y_pos <= 120 + (75 * i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (730, 80 + (i * 75)), (950, 120 + (75 * i)), line_color1, -1)
                    if (finger1_x_pos >= 490 and finger1_x_pos <= 710):
                        if (finger1_y_pos >= (80 + (i * 75)) and finger1_y_pos <= 120 + (75 * i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (490, 80 + (i * 75)), (710, 120 + (75 * i)), line_color1, -1)
                    if (finger1_x_pos >= 250 and finger1_x_pos <= 470):
                        if (finger1_y_pos >= (80 + (i * 75)) and finger1_y_pos <= 120 + (75 * i)):
                            box_dect[3][i] = 1
                            cv2.rectangle(image, (250, 80 + (i * 75)), (470, 120 + (75 * i)), line_color1, -1)
                    if (finger1_x_pos >= 9 and finger1_x_pos <= 230):
                        if (finger1_y_pos >= (80 + (i * 75)) and finger1_y_pos <= 120 + (75 * i)):
                            box_dect[4][i] = 1
                            cv2.rectangle(image, (9, 80 + (i * 75)), (230, 120 + (75 * i)), line_color1, -1)
                    # 중지
                    if (finger3_x_pos >= 970 and finger3_x_pos <= 1190):
                        if (finger3_y_pos >= (80 + (i * 75)) and finger3_y_pos <= 120 + (75 * i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (970, 80 + (i * 75)), (1190, 120 + (75 * i)), line_color1, -1)
                    if (finger3_x_pos >= 730 and finger3_x_pos <= 950):
                        if (finger3_y_pos >= (80 + (i * 75)) and finger3_y_pos <= 120 + (75 * i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (730, 80 + (i * 75)), (950, 120 + (75 * i)), line_color1, -1)
                    if (finger3_x_pos >= 490 and finger3_x_pos <= 710):
                        if (finger3_y_pos >= (80 + (i * 75)) and finger3_y_pos <= 120 + (75 * i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (490, 80 + (i * 75)), (710, 120 + (75 * i)), line_color1, -1)
                    if (finger3_x_pos >= 250 and finger3_x_pos <= 470):
                        if (finger3_y_pos >= (80 + (i * 75)) and finger3_y_pos <= 120 + (75 * i)):
                            box_dect[3][i] = 1
                            cv2.rectangle(image, (250, 80 + (i * 75)), (470, 120 + (75 * i)), line_color1, -1)
                    if (finger3_x_pos >= 9 and finger3_x_pos <= 230):
                        if (finger3_y_pos >= (80 + (i * 75)) and finger3_y_pos <= 120 + (75 * i)):
                            box_dect[4][i] = 1
                            cv2.rectangle(image, (9, 80 + (i * 75)), (230, 120 + (75 * i)), line_color1, -1)
                    # 약지
                    if (finger4_x_pos >= 970 and finger4_x_pos <= 1190):
                        if (finger4_y_pos >= (80 + (i * 75)) and finger4_y_pos <= 120 + (75 * i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (970, 80 + (i * 75)), (1190, 120 + (75 * i)), line_color1, -1)
                    if (finger4_x_pos >= 730 and finger4_x_pos <= 950):
                        if (finger4_y_pos >= (80 + (i * 75)) and finger4_y_pos <= 120 + (75 * i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (730, 80 + (i * 75)), (950, 120 + (75 * i)), line_color1, -1)
                    if (finger4_x_pos >= 490 and finger4_x_pos <= 710):
                        if (finger4_y_pos >= (80 + (i * 75)) and finger4_y_pos <= 120 + (75 * i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (490, 80 + (i * 75)), (710, 120 + (75 * i)), line_color1, -1)
                    if (finger4_x_pos >= 250 and finger4_x_pos <= 470):
                        if (finger4_y_pos >= (80 + (i * 75)) and finger4_y_pos <= 120 + (75 * i)):
                            box_dect[3][i] = 1
                            cv2.rectangle(image, (250, 80 + (i * 75)), (470, 120 + (75 * i)), line_color1, -1)
                    if (finger4_x_pos >= 9 and finger4_x_pos <= 230):
                        if (finger4_y_pos >= (80 + (i * 75)) and finger4_y_pos <= 120 + (75 * i)):
                            box_dect[4][i] = 1
                            cv2.rectangle(image, (9, 80 + (i * 75)), (230, 120 + (75 * i)), line_color1, -1)
                    # 소지
                    if (finger5_x_pos >= 970 and finger5_x_pos <= 1190):
                        if (finger5_y_pos >= (80 + (i * 75)) and finger5_y_pos <= 120 + (75 * i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (970, 80 + (i * 75)), (1190, 120 + (75 * i)), line_color1, -1)
                    if (finger5_x_pos >= 730 and finger5_x_pos <= 950):
                        if (finger5_y_pos >= (80 + (i * 75)) and finger5_y_pos <= 120 + (75 * i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (730, 80 + (i * 75)), (950, 120 + (75 * i)), line_color1, -1)
                    if (finger5_x_pos >= 490 and finger5_x_pos <= 710):
                        if (finger5_y_pos >= (80 + (i * 75)) and finger5_y_pos <= 120 + (75 * i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (490, 80 + (i * 75)), (710, 120 + (75 * i)), line_color1, -1)
                    if (finger5_x_pos >= 250 and finger5_x_pos <= 470):
                        if (finger5_y_pos >= (80 + (i * 75)) and finger5_y_pos <= 120 + (75 * i)):
                            box_dect[3][i] = 1
                            cv2.rectangle(image, (250, 80 + (i * 75)), (470, 120 + (75 * i)), line_color1, -1)
                    if (finger5_x_pos >= 9 and finger5_x_pos <= 230):
                        if (finger5_y_pos >= (80 + (i * 75)) and finger5_y_pos <= 120 + (75 * i)):
                            box_dect[4][i] = 1
                            cv2.rectangle(image, (9, 80 + (i * 75)), (230, 120 + (75 * i)), line_color1, -1)

                if lmList != []:
                    # getting the value at a point
                    # x      #y
                    x1, y1 = lmList[4][1], lmList[4][2]  # thumb
                    x2, y2 = lmList[8][1], lmList[8][2]  # index finger
                    # creating circle at the tips of thumb and index finger
                    cv2.circle(image, (x1, y1), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
                    cv2.circle(image, (x2, y2), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0),
                             3)  # create a line b/w tips of index finger and thumb

                    length = hypot(x2 - x1, y2 - y1)  # distance b/w tips using hypotenuse
                    # from numpy we find our length,by converting hand range in terms of volume range ie b/w -63.5 to 0

                    # Hand range 30 - 350
                    # Volume range -63.5 - 0.0
                    # creating volume bar for volume level
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 255),
                                  4)  # vid ,initial position ,ending position ,rgb ,thickness

                    # tell the volume percentage ,location,font of text,length,rgb color,thickness

                hand = mp_drawing.draw_landmarks(  # 손가락 뼈대 그리기
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # if box_dect[1][2] == 1 and (box_dect[1][4]) == 1:
        #   print('코드 A7가 감지되었습니다.')
        #  test_sound = pygame.mixer.Sound('ele_E.wav')
        # test_sound.play(1)
        # cv2.putText(image, "A7", (800,700 ), cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 5)

        if box_dect[0][0] and box_dect[1][4] and box_dect[2][5]:
            print('코드 C가 감지되었습니다.')

        if box_dect[0][1] == 1 and (box_dect[1][2]) == 1:
            if box_dect[1][3] and box_dect[2][0]:
                print('코드 Am7이 감지되었습니다.')

            elif box_dect[1][0]:
                print('코드 D7가 감지되었습니다.')

            elif box_dect[0][0] and box_dect[0][5] and box_dect[2][3] and box_dect[2][4]:
                print('코드 F가 감지되었습니다.')

        if box_dect[2][5] and box_dect[2][4]:
            print('코드 G7가 감지되었습니다.')

        if box_dect[2][3] and box_dect[2][1]:
            print('코드 C7가 감지되었습니다.')

        if box_dect[1][3]:
            print('코드 Am가 감지되었습니다.')

        if box_dect[1][2] and box_dect[1][4]:
            if box_dect[1][3]:
                print('코드 A가 감지되었습니다.')

            else:
                print('코드 A7가 감지되었습니다.')

        if (box_dect[1][2]) == 1 and box_dect[2][1] == 1 and box_dect[0][4] == 0:
            print('코드 CM7가 감지되었습니다.')

        if box_dect[1][0] == 1 and (box_dect[1][2]) == 1 and box_dect[2][1] == 1:
            print('코드 D가 감지되었습니다.')

        if (box_dect[1][1]) == 1 and box_dect[2][0] == 1:
            if box_dect[2][5]:
                print('코드 G가 감지되었습니다.')

        if box_dect[0][5] == 1 and (box_dect[1][1]) == 1 and box_dect[2][0] == 1:
            print('코드 G#7가 감지되었습니다.')

        if box_dect[1][1] == 1 and (box_dect[2][5]) == 1 and box_dect[3][2] == 1 and box_dect[3][3] == 1 and \
                box_dect[3][4]:
            print('코드 B가 감지되었습니다.')

        if box_dect[0][2] == 1 and (box_dect[1][3]) == 1 and box_dect[1][4] == 1:
            print('코드 E가 감지되었습니다.')

        if box_dect[2][0] == 1 and (box_dect[2][1]) == 1 and box_dect[2][5] == 1 and box_dect[3][4] == 1 and \
                box_dect[4][2] and box_dect[4][3]:
            print('코드 Cm가 감지되었습니다.')

        if box_dect[1][1] and box_dect[1][2]:
            print('코드 Em가 감지되었습니다.')

        box_dect = [[0] * 6 for _ in range(5)]  # 인식 후에 2차원 리스트 초기화를 통해 다른거 인식할 준비

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:  # esc누르면 꺼짐
            break
    cap.release()
    pygame.quit()