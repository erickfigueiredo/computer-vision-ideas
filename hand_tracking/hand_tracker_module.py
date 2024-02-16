import cv2
import time
from enum import Enum
import mediapipe as mp


class Keys(Enum):
    ESC = 27
    SPACE = 32
    TAB = 9


class HandTracker:
    """
    Class for tracking hands using the MediaPipe library.
    """

    def __init__(self, mode=False, max_hands=2, min_detection_confidence=.5, min_tracking_confidence=.5):
        self.parameters = {
            'static_image_mode': mode,
            'max_num_hands': max_hands,
            'min_detection_confidence': min_detection_confidence,
            'min_tracking_confidence': min_tracking_confidence
        }

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(**self.parameters)

        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_id=0, draw=True):
        landmarks = []

        if self.results.multi_hand_landmarks:
            if hand_id > len(self.results.multi_hand_landmarks)-1:
                raise Exception('Hand ID out of range!')
            for i, lm in enumerate(self.results.multi_hand_landmarks[hand_id].landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                landmarks.append({
                    'id': i,
                    'x': cx,
                    'y': cy
                })

                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return landmarks


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    curr_time, prev_time = 0, 0

    draw_skeleton = True
    draw_points = True

    while True:
        success, img = cap.read()

        if not success:
            raise Exception('Error reading the frame!')

        key = cv2.waitKey(1)

        if key == Keys.ESC.value:
            break

        elif key == Keys.SPACE.value:
            draw_skeleton = not draw_skeleton
        
        elif key == Keys.TAB.value:
            draw_points = not draw_points


        img = tracker.find_hands(img, draw_skeleton)


        tracker.find_position(img, draw=draw_points)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f'FPS: {round(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 250, 0), 2)

        cv2.imshow('Hand Tracker', img)


if __name__ == '__main__':
    main()
