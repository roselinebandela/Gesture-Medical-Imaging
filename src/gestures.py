import math
import time
from collections import deque

WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


class GestureDetector:
    def __init__(self):
        self.swipe_cooldown = 0.4
        self.last_swipe_time = 0.0

        self.position_history = deque(maxlen=10)
        self.pose_history = deque(maxlen=5)
        self.stable_pose = "NO HAND"
        self._pose_required_streak = 3
        self._was_pinching = False

    def _coords(self, landmarks, index):
        for lm in landmarks:
            if lm["id"] == index:
                return (lm["x"], lm["y"])
        return None

    def _distance(self, p1, p2):
        if not p1 or not p2:
            return float("inf")
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _hand_scale(self, landmarks):
        """Reference unit for this hand: wrist-to-middle-knuckle distance."""
        wrist = self._coords(landmarks, WRIST)
        middle_mcp = self._coords(landmarks, MIDDLE_MCP)
        scale = self._distance(wrist, middle_mcp)
        return scale if scale and scale != float("inf") else 1.0

    def _finger_extended(self, landmarks, wrist, tip_id, pip_id, margin_ratio, scale):
        tip = self._coords(landmarks, tip_id)
        pip = self._coords(landmarks, pip_id)
        if not tip or not pip:
            return False
        return self._distance(tip, wrist) > self._distance(pip, wrist) + margin_ratio * scale

    def fingers_up(self, landmarks):
        """Returns [thumb, index, middle, ring, pinky] as 1/0."""
        if len(landmarks) < 21:
            return [0, 0, 0, 0, 0]

        wrist = self._coords(landmarks, WRIST)
        scale = self._hand_scale(landmarks)
        margin = 0.1

        pinky_mcp = self._coords(landmarks, PINKY_MCP)
        thumb_tip = self._coords(landmarks, THUMB_TIP)
        thumb_mcp = self._coords(landmarks, THUMB_MCP)
        thumb_up = 0
        if thumb_tip and thumb_mcp and pinky_mcp:
            thumb_up = int(self._distance(thumb_tip, pinky_mcp) >
                            self._distance(thumb_mcp, pinky_mcp) + margin * scale)

        fingers = [thumb_up]
        for tip_id, pip_id in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                                (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)):
            fingers.append(int(self._finger_extended(landmarks, wrist, tip_id, pip_id, margin, scale)))
        return fingers

    def is_pinching(self, landmarks):
        """Pure current-frame state: are thumb+index tips touching right now?
        Use this for continuous actions (e.g. zoom while held)."""
        if len(landmarks) < 21:
            return False
        scale = self._hand_scale(landmarks)
        thumb = self._coords(landmarks, THUMB_TIP)
        index = self._coords(landmarks, INDEX_TIP)
        return self._distance(thumb, index) < 0.45 * scale

    def detect_pinch_trigger(self, landmarks):
        """Fires once on the rising edge of a pinch (not-pinching -> pinching).
        Use this for discrete one-shot actions (e.g. placing a marker)."""
        pinching = self.is_pinching(landmarks)
        triggered = pinching and not self._was_pinching
        self._was_pinching = pinching
        return triggered

    def classify_pose(self, landmarks):
        if len(landmarks) < 21:
            return "NO HAND"

        fingers = self.fingers_up(landmarks)
        count = sum(fingers)
        thumb, index, middle, ring, pinky = fingers

        if count >= 4:
            return "OPEN PALM"
        if count == 0:
            return "FIST"
        if fingers == [0, 1, 0, 0, 0]:
            return "INDEX POINTING"
        if fingers in ([0, 1, 1, 0, 0], [1, 1, 1, 0, 0]):
            return "PEACE"
        return "OTHER"

    def detect_hand_pose(self, landmarks):
        """Debounced pose: only changes after the same pose repeats a few frames."""
        raw_pose = self.classify_pose(landmarks)
        self.pose_history.append(raw_pose)

        if len(self.pose_history) >= self._pose_required_streak:
            recent = list(self.pose_history)[-self._pose_required_streak:]
            if all(p == raw_pose for p in recent):
                self.stable_pose = raw_pose

        return self.stable_pose

    def detect_swipe(self, landmarks):
        if len(landmarks) < 21:
            return None

        now = time.time()
        if now - self.last_swipe_time < self.swipe_cooldown:
            return None

        index_tip = self._coords(landmarks, INDEX_TIP)
        scale = self._hand_scale(landmarks)
        if not index_tip:
            return None

        self.position_history.append((index_tip[0], now))
        if len(self.position_history) < 3:
            return None

        start_x, start_t = self.position_history[0]
        end_x, end_t = self.position_history[-1]
        dx = end_x - start_x

        if end_t - start_t > 1.2:
            self.position_history.clear()
            return None

        if abs(dx) > 1.4 * scale:
            self.last_swipe_time = now
            self.position_history.clear()
            return "RIGHT" if dx > 0 else "LEFT"

        return None
