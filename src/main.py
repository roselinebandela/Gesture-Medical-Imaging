"""Gesture-controlled chest X-ray viewer with real AI pathology detection.

Touchless navigation via webcam hand gestures (see print_controls below),
paired with an actual pretrained chest X-ray classifier and Grad-CAM heatmap
from ai_engine.py. This is a research/demo tool, not a diagnostic device.
"""
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import ui_render as ui
from ai_engine import AIEngine
from gestures import GestureDetector
from hand_tracker import HandTracker
from image_viewer import MedicalImageViewer

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "screenshots"

XRAY_WINDOW = "X-Ray Viewer"
CAMERA_WINDOW = "Gesture Camera"

COLORS = {
    "bg": (18, 16, 24),
    "panel": (24, 22, 32),
    "accent": (255, 180, 80),
    "danger": (70, 70, 235),
    "warning": (60, 170, 235),
    "mild": (150, 200, 90),
    "success": (140, 210, 120),
    "text": (240, 240, 245),
    "text_dim": (155, 155, 168),
    "divider": (55, 52, 68),
}

ZOOM_STEP_IN = 1.045
ZOOM_STEP_OUT = 1 / 1.045


class MedicalImagingSystem:
    def __init__(self):
        print("Gesture-Controlled Chest X-Ray Viewer\n")

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.tracker = HandTracker()
        self.detector = GestureDetector()
        self.viewer = MedicalImageViewer(viewport_size=(860, 860))
        self.ai = AIEngine()

        self.active = True
        self.show_heatmap = True
        self.marker_mode = False
        self.ai_result = None
        self.analyzed_index = None
        self.glow_pulse = 0.0
        self.last_screenshot_msg = None
        self.last_screenshot_time = 0.0

        cv2.namedWindow(XRAY_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(XRAY_WINDOW, *self.viewer.viewport_size)
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW, 1280, 720)

        self.print_controls()
        self.refresh_analysis()

    def print_controls(self):
        print("Gestures:")
        print("  Open palm         activate")
        print("  Fist              deactivate")
        print("  Pinch (hold)      zoom in, or place a marker in marker mode")
        print("  Peace sign (hold) zoom out")
        print("  Point + hold      pan around when zoomed in")
        print("  Swipe left/right  previous / next image")
        print()
        print("Keys:")
        print("  M  marker mode      Z  undo last marker   C  clear markers")
        print("  H  toggle heatmap   I  invert              R  reset view")
        print("  [ ]  brightness     - =  contrast           S  screenshot   Q  quit")
        print()

    def refresh_analysis(self):
        """Analyze the current image once and cache by index — the model is
        deterministic, so re-running on an unchanged image wastes CPU."""
        if self.analyzed_index == self.viewer.index:
            return
        self.ai_result = self.ai.analyze(self.viewer.image)
        self.analyzed_index = self.viewer.index

    # ------------------------------------------------------------------ UI

    def draw_medical_overlay(self, base_image):
        display = base_image.copy()
        h, w = display.shape[:2]
        result = self.ai_result

        # Slim top strip: image position + view-state badges only.
        display = ui.draw_panel(display, 0, 0, w, 34, COLORS["bg"], 0.45)
        if self.viewer.images:
            counter = f"{self.viewer.index + 1} / {len(self.viewer.images)}"
            display = ui.put_text(display, counter, (14, 8), size=14, color=COLORS["text_dim"])

        badges = []
        if self.viewer.zoom > 1.02:
            badges.append(f"Zoom {self.viewer.zoom:.1f}x")
        if self.viewer.inverted:
            badges.append("Inverted")
        if self.viewer.brightness or self.viewer.contrast != 1.0:
            badges.append("Adjusted")
        if self.marker_mode:
            badges.append("Marker mode")
        badge_text = "   ".join(badges)
        if badge_text:
            tw, _ = ui.text_size(badge_text, 14)
            display = ui.put_text(display, badge_text, (w - tw - 14, 8), size=14, color=COLORS["accent"])

        # Bottom info card: the finding, or a quiet "nothing stood out" chip.
        bar_h = 130
        bar_y = h - bar_h
        display = ui.draw_rounded_panel(display, 14, bar_y, w - 28, bar_h - 14,
                                         COLORS["panel"], alpha=0.72, radius=14)

        if result is None or result.error:
            msg = result.error if (result and result.error) else "Analyzing..."
            display = ui.put_text(display, msg, (32, bar_y + 20), size=14, color=COLORS["text_dim"])
        elif result.is_normal:
            display = ui.draw_dot(display, (32, bar_y + 28), 5, COLORS["success"])
            display = ui.put_text(display, "No standout finding", (48, bar_y + 18),
                                   size=17, color=COLORS["success"], weight="semibold")
            display = ui.put_text(display, "No pathology scored clearly above this image's own baseline.",
                                   (32, bar_y + 46), size=12, color=COLORS["text_dim"])
        else:
            finding = result.primary
            if finding.probability > 0.65:
                sev_color, sev_text = COLORS["danger"], "ELEVATED"
            elif finding.probability > 0.55:
                sev_color, sev_text = COLORS["warning"], "MODERATE"
            else:
                sev_color, sev_text = COLORS["mild"], "MILD"

            display = ui.draw_dot(display, (32, bar_y + 28), 5, sev_color)
            label = finding.condition.replace("_", " ")
            display = ui.put_text(display, label, (48, bar_y + 18),
                                   size=18, color=COLORS["text"], weight="semibold")

            chip_text = sev_text
            chip_w, _ = ui.text_size(chip_text, 12, "semibold")
            chip_x = w - chip_w - 40
            display = ui.draw_rounded_rect(display, chip_x, bar_y + 14, chip_w + 16, 22,
                                            sev_color, thickness=-1, radius=10)
            display = ui.put_text(display, chip_text, (chip_x + 8, bar_y + 17),
                                   size=12, color=(15, 15, 20), weight="semibold")

            if finding.description:
                display = ui.put_text(display, finding.description, (32, bar_y + 46),
                                       size=12, color=COLORS["text_dim"])

            bar_x, bar_yy, bw, bh = 32, bar_y + 70, w - 64 - 90, 8
            cv2.rectangle(display, (bar_x, bar_yy), (bar_x + bw, bar_yy + bh), (45, 42, 55), -1)
            fill = int(bw * min(finding.probability, 1.0))
            cv2.rectangle(display, (bar_x, bar_yy), (bar_x + fill, bar_yy + bh), sev_color, -1)
            display = ui.put_text(display, f"{finding.probability * 100:.0f}%",
                                   (bar_x + bw + 12, bar_yy - 5), size=13, color=COLORS["text"])

            if len(result.findings) > 1:
                others = ", ".join(f.condition.replace("_", " ") for f in result.findings[1:4])
                display = ui.put_text(display, f"Also elevated: {others}", (32, bar_y + 96),
                                       size=11, color=COLORS["text_dim"])

        if time.time() - self.last_screenshot_time < 1.6 and self.last_screenshot_msg:
            msg_w, _ = ui.text_size(self.last_screenshot_msg, 13, "semibold")
            display = ui.draw_rounded_panel(display, w - msg_w - 40, 40, msg_w + 26, 30,
                                             (20, 60, 30), alpha=0.85, radius=8)
            display = ui.put_text(display, self.last_screenshot_msg, (w - msg_w - 27, 47),
                                   size=13, color=COLORS["success"], weight="semibold")

        return display

    def draw_gesture_ui(self, frame, gesture_text):
        h, w = frame.shape[:2]
        frame = ui.draw_panel(frame, 0, 0, w, 50, COLORS["bg"], 0.4)

        status_color = COLORS["success"] if self.active else COLORS["danger"]
        if self.active:
            self.glow_pulse = (self.glow_pulse + 0.05) % (2 * np.pi)
            glow = int(7 + 3 * np.sin(self.glow_pulse))
            cv2.circle(frame, (26, 25), glow, status_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (26, 25), 7, status_color, -1, cv2.LINE_AA)

        status = "ACTIVE" if self.active else "STANDBY"
        frame = ui.put_text(frame, status, (44, 13), size=16, color=status_color, weight="semibold")
        frame = ui.put_text(frame, gesture_text, (170, 14), size=14, color=COLORS["text_dim"])

        frame = ui.draw_panel(frame, 0, h - 32, w, 32, COLORS["bg"], 0.4)
        frame = ui.put_text(
            frame,
            "PALM activate   FIST pause   PINCH zoom/mark   PEACE zoom out   "
            "POINT pan   SWIPE change image",
            (14, h - 24), size=12, color=COLORS["text_dim"])
        return frame

    def save_screenshot(self, medical_image):
        try:
            SCREENSHOTS_DIR.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = SCREENSHOTS_DIR / f"xray_capture_{timestamp}.png"
            ok = cv2.imwrite(str(filename), medical_image)
            if not ok:
                raise IOError(f"cv2.imwrite reported failure for {filename}")
            print(f"Screenshot saved: {filename}")
            self.last_screenshot_msg = "Saved"
        except Exception as exc:
            print(f"Screenshot failed: {type(exc).__name__}: {exc}")
            self.last_screenshot_msg = "Save failed - see console"
        self.last_screenshot_time = time.time()

    # ----------------------------------------------------------------- run

    def run(self):
        print("Starting camera...")
        try:
            while True:
                success, frame = self.camera.read()
                if not success:
                    print("Camera not available.")
                    break

                frame = cv2.flip(frame, 1)
                frame_shape = frame.shape[:2]

                frame, landmarks, handedness, cursor_pos = self.tracker.find_hands(frame)

                if landmarks:
                    pose = self.detector.detect_hand_pose(landmarks)
                    pinching = self.detector.is_pinching(landmarks)
                    pinch_trigger = self.detector.detect_pinch_trigger(landmarks)
                    swipe = self.detector.detect_swipe(landmarks)
                else:
                    pose = "NO HAND"
                    pinching = False
                    pinch_trigger = False
                    swipe = None
                    cursor_pos = None

                gesture_text = pose

                if pose == "OPEN PALM":
                    self.active = True
                    gesture_text = "SYSTEM ACTIVE"
                elif pose == "FIST":
                    self.active = False
                    gesture_text = "SYSTEM PAUSED"

                if self.active:
                    self.viewer.update_cursor_position(cursor_pos, frame_shape)
                    cursor_img = self.viewer.cursor_on_image

                    if self.marker_mode:
                        gesture_text = "MARKER MODE"
                        if pinch_trigger:
                            self.viewer.add_annotation()
                            gesture_text = "POINT MARKED"
                    else:
                        if pinching:
                            self.viewer.zoom_by(ZOOM_STEP_IN, anchor=cursor_img)
                            gesture_text = "ZOOM IN"
                        elif pose == "PEACE":
                            self.viewer.zoom_by(ZOOM_STEP_OUT)
                            gesture_text = "ZOOM OUT"
                        elif pose == "INDEX POINTING" and self.viewer.zoom > 1.02 and cursor_img:
                            self.viewer.pan_toward(cursor_img, ease=0.06)
                            gesture_text = "PANNING"

                    if swipe == "RIGHT":
                        self.viewer.next_image()
                        gesture_text = "NEXT IMAGE"
                    elif swipe == "LEFT":
                        self.viewer.previous_image()
                        gesture_text = "PREVIOUS IMAGE"
                else:
                    self.viewer.cursor_on_image = None

                self.refresh_analysis()

                heatmap_base = None
                if self.show_heatmap and self.ai_result and self.ai_result.heatmap_overlay is not None:
                    heatmap_base = self.ai_result.heatmap_overlay

                medical_image = self.viewer.show(base_override=heatmap_base)
                medical_display = self.draw_medical_overlay(medical_image)
                frame = self.draw_gesture_ui(frame, gesture_text)

                cv2.imshow(CAMERA_WINDOW, frame)
                cv2.imshow(XRAY_WINDOW, medical_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("m"):
                    self.marker_mode = not self.marker_mode
                elif key == ord("c"):
                    self.viewer.clear_annotations()
                elif key == ord("z"):
                    self.viewer.undo_annotation()
                elif key == ord("h"):
                    self.show_heatmap = not self.show_heatmap
                elif key == ord("i"):
                    self.viewer.toggle_invert()
                elif key == ord("r"):
                    self.viewer.reset_view()
                elif key == ord("["):
                    self.viewer.adjust_brightness(-10)
                elif key == ord("]"):
                    self.viewer.adjust_brightness(10)
                elif key == ord("-"):
                    self.viewer.adjust_contrast(1 / 1.1)
                elif key == ord("="):
                    self.viewer.adjust_contrast(1.1)
                elif key == ord("s"):
                    self.save_screenshot(medical_display)

        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    MedicalImagingSystem().run()
