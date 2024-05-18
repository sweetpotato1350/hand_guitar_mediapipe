import cv2  # opencv라이브러리 import
import mediapipe as mp  # MediaPipe패키지 import하고 mp로 사용
import numpy
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

# Load the sound files
sound_files = {
    'A7': 'sounds/A7.mp3',
    'Am7': 'sounds/Am7.mp3',
    'Am': 'sounds/Am.mp3',
    'C': 'sounds/C.mp3',
    'C7': 'sounds/C7.mp3',
    'Cm': 'sounds/Cm.mp3',
    'CM7': 'sounds/CM7..mp3',
    'D7': 'sounds/D7.mp3',
    'D': 'sounds/D.mp3',
    'Em': 'sounds/Em.mp3',
    'F': 'sounds/F.mp3',
    'G7': 'soundsG7.mp3',
    'G': 'sounds/G.mp3'
}

# mediapipe 패키지에서 사용할 기능들
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  # 손 인식을 위한 객체
box_dect = [[0] * 6 for _ in range(3)]  # 인식 후에 2차원 리스트 초기화를 통해 다른거 인식할 준비

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

        line_color = (10, 10, 10, 255)
        line_color1 = (255, 255, 255)
        for hor_line_gen in range(6):
            cv2.line(image, (0, 100+75*hor_line_gen), (1500, 100+75*hor_line_gen), line_color, 3)  # Line 생성(가로줄)
            cv2.rectangle(image, (810, 80+75*hor_line_gen), (1190, 120+75*hor_line_gen), (255, 0, 0), 6) #box 1
            cv2.rectangle(image, (410, 80+75*hor_line_gen), (790, 120+75*hor_line_gen), (0, 255, 0), 6)  #box 2
            cv2.rectangle(image, (390, 80+75*hor_line_gen), (9, 120+75*hor_line_gen), (0, 0, 255), 6)  # box 3

        cv2.line(image, (800, 100), (800, 475), line_color, 3)  # vertical Line 1
        cv2.line(image, (400, 100), (400, 475), line_color, 3)  # vertical Line 2

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

                for i in range(6):
                    #검지
                    if (finger2_x_pos >= 810 and finger2_x_pos <= 1190):
                        if (finger2_y_pos >= (80+(i*75)) and finger2_y_pos <= 120+(75*i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (810, 80+(i*75)), (1190, 120+(75*i)), line_color1, -1)
                    if (finger2_x_pos >= 410 and finger2_x_pos <= 790):
                        if (finger2_y_pos >= (80+(i*75)) and finger2_y_pos <= 120+(75*i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (410, 80+(i*75)), (790, 120+(75*i)), line_color1, -1)
                    if (finger2_x_pos >= 9 and finger2_x_pos <= 390):
                        if (finger2_y_pos >= (80+(i*75)) and finger2_y_pos <= 120+(75*i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (9, 80+(i*75)), (390, 120+(75*i)), line_color1, -1)
                    #엄지
                    if (finger1_x_pos >= 810 and finger1_x_pos <= 1190):
                        if (finger1_y_pos >= (80+(i*75)) and finger1_y_pos <= 120+(75*i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (810, 80+(i*75)), (1190, 120+(75*i)), line_color1, -1)
                    if (finger1_x_pos >= 410 and finger1_x_pos <= 790):
                        if (finger1_y_pos >= (80+(i*75)) and finger1_y_pos <= 120+(75*i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (410, 80+(i*75)), (790, 120+(75*i)), line_color1, -1)
                    if (finger1_x_pos >= 9 and finger1_x_pos <= 390):
                        if (finger1_y_pos >= (80+(i*75)) and finger1_y_pos <= 120+(75*i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (9, 80+(i*75)), (390, 120+(75*i)), line_color1, -1)
                    # 중지
                    if (finger3_x_pos >= 810 and finger3_x_pos <= 1190):
                        if (finger3_y_pos >= (80+(i*75)) and finger3_y_pos <= 120+(75*i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (810, 80+(i*75)), (1190, 120+(75*i)), line_color1, -1)
                    if (finger3_x_pos >= 410 and finger3_x_pos <= 790):
                        if (finger3_y_pos >= (80+(i*75)) and finger3_y_pos <= 120+(75*i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (410, 80+(i*75)), (790, 120+(75*i)), line_color1, -1)
                    if (finger3_x_pos >= 9 and finger3_x_pos <= 390):
                        if (finger3_y_pos >= (80+(i*75)) and finger3_y_pos <= 120+(75*i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (9, 80+(i*75)), (390, 120+(75*i)), line_color1, -1)
                    # 약지
                    if (finger4_x_pos >= 810 and finger4_x_pos <= 1190):
                        if (finger4_y_pos >= (80+(i*75)) and finger4_y_pos <= 120+(75*i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (810, 80+(i*75)), (1190, 120+(75*i)), line_color1, -1)
                    if (finger4_x_pos >= 410 and finger4_x_pos <= 790):
                        if (finger4_y_pos >= (80+(i*75)) and finger4_y_pos <= 120+(75*i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (410, 80+(i*75)), (790, 120+(75*i)), line_color1, -1)
                    if (finger4_x_pos >= 9 and finger4_x_pos <= 390):
                        if (finger4_y_pos >= (80+(i*75)) and finger4_y_pos <= 120+(75*i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (9, 80+(i*75)), (390, 120+(75*i)), line_color1, -1)
                    #소지
                    if (finger5_x_pos >= 810 and finger5_x_pos <= 1190):
                        if (finger5_y_pos >= (80+(i*75)) and finger5_y_pos <= 120+(75*i)):
                            box_dect[0][i] = 1
                            cv2.rectangle(image, (810, 80+(i*75)), (1190, 120+(75*i)), line_color1, -1)
                    if (finger5_x_pos >= 410 and finger5_x_pos <= 790):
                        if (finger5_y_pos >= (80+(i*75)) and finger5_y_pos <= 120+(75*i)):
                            box_dect[1][i] = 1
                            cv2.rectangle(image, (410, 80+(i*75)), (790, 120+(75*i)), line_color1, -1)
                    if (finger5_x_pos >= 9 and finger5_x_pos <= 390):
                        if (finger5_y_pos >= (80+(i*75)) and finger5_y_pos <= 120+(75*i)):
                            box_dect[2][i] = 1
                            cv2.rectangle(image, (9, 80+(i*75)), (390, 120+(75*i)), line_color1, -1)
                #print(box_dect)
                # box_dect[0] 은 가장 왼쪽줄, 1은 중간줄, 2는 오른쪽 줄을 감지하고
                # box_dect[0][0]은 가장 왼쪽 줄에서 가장 윗 박스가 감지되면 1로 셋팅되도록 되어 있음

                box_dect=numpy.array(box_dect)
                '''if (sum(box_dect[1][1:4]) == 2): # 이런식으로 코드 표를 조합하면 됨
                        print('코드 A7가 감지되었습니다.')'''

                if box_dect[1][2] == 1and (box_dect[1][4])==1:
                    sound_file = sound_files['A7']
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                    print('코드 A7이 감지되었습니다.')
                    break
                if box_dect[0][4] == 1and (box_dect[1][2])==1:
                    sound_file = sound_files['Am7']
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                    print('코드 Am7이 감지되었습니다.')
                    break
                if box_dect[0][4] == 1 and (box_dect[1][2]) == 1and box_dect[1][3] == 1:
                    sound_file = sound_files['Am']
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                    print('코드 Am가 감지되었습니다.')
                    break
                if box_dect[0][4] == 1and (box_dect[1][2])==1and box_dect[2][1]:
                    sound_file = sound_files['C']
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                    print('코드 C가 감지되었습니다.')
                    break
                if box_dect[0][4] == 1and (box_dect[1][2])==1and box_dect[2][1] and box_dect[2][3]:
                    sound_file = sound_files['C7']
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                    print('코드 C7이 감지되었습니다.')
                    break
                if (box_dect[1][2])==1and box_dect[2][1] :
                    sound_file = sound_files['CM7']
                    sound = pygame.mixer.Sound(sound_file)
                    sound.play()
                    print('코드 CM7가 감지되었습니다.')
                    break
                box_dect = [[0] * 6 for _ in range(3)]  # 인식 후에 2차원 리스트 초기화를 통해 다른거 인식할 준비

                hand = mp_drawing.draw_landmarks(  # 손가락 뼈대 그리기
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:  # esc누르면 꺼짐
            break
cap.release()
cv2.destroyAllWindows()