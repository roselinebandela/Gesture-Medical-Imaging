# hand_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Smooth cursor with moving average
        self.cursor_history = deque(maxlen=5)
        
        # Drawing styles
        self.landmark_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)
        self.connection_style = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=1)
        
    def smooth_cursor(self, raw_x, raw_y):
        """Smooth cursor movement"""
        self.cursor_history.append((raw_x, raw_y))
        
        if len(self.cursor_history) < 2:
            return raw_x, raw_y
        
        # Weighted average
        weights = np.exp(np.linspace(-1, 0, len(self.cursor_history)))
        weights /= weights.sum()
        
        smooth_x = sum(p[0] * w for p, w in zip(self.cursor_history, weights))
        smooth_y = sum(p[1] * w for p, w in zip(self.cursor_history, weights))
        
        return int(smooth_x), int(smooth_y)
    
    def find_hands(self, frame):
        if frame is None:
            return frame, [], None, None
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        landmarks = []
        handedness = None
        cursor_pos = None
        
        if results.multi_hand_landmarks:
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_style, self.connection_style)
                
                h, w, _ = frame.shape
                
                for idx, lm in enumerate(hand_landmarks.landmark):
                    landmarks.append({
                        'id': idx,
                        'x': int(lm.x * w),
                        'y': int(lm.y * h),
                        'z': lm.z
                    })
                
                # Index finger tip
                index_tip = hand_landmarks.landmark[8]
                raw_x = int(index_tip.x * w)
                raw_y = int(index_tip.y * h)
                
                # Smooth cursor
                smooth_x, smooth_y = self.smooth_cursor(raw_x, raw_y)
                cursor_pos = (smooth_x, smooth_y)
                
                # Draw cursor
                if cursor_pos:
                    cx, cy = cursor_pos
                    # Crosshair
                    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 2)
                    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                    # Raw position dot
                    cv2.circle(frame, (raw_x, raw_y), 4, (0, 255, 0), -1)
        
        return frame, landmarks, handedness, cursor_pos