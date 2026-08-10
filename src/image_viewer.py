# image_viewer.py
"""Chest X-ray viewer: fixed-size viewport with real pan/zoom.

Zoom no longer resizes the whole image array (which, for the ~2000px source
photos this project uses, made the OpenCV window balloon to a size larger
than most screens — the real cause zoom "didn't work"). The displayed image
is always exactly `viewport_size`; zooming crops and scales a region of the
original image into that fixed frame, centered on a pan point.
"""
import cv2
import numpy as np
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"

MIN_ZOOM = 1.0
MAX_ZOOM = 6.0
MARKER_COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


class MedicalImageViewer:
    def __init__(self, viewport_size=(880, 880)):
        self.viewport_size = viewport_size
        self.images = []
        self.index = 0
        self.zoom = MIN_ZOOM
        self.pan = None  # (x, y) in original-image pixel coords; None = centered
        self.cursor_on_image = None
        self.annotations = []
        self.brightness = 0    # -100..100
        self.contrast = 1.0    # 0.4..3.0
        self.inverted = False

        self.load_images()
        if self.images:
            self.load_current()
        else:
            self.image = np.ones((512, 512, 3), dtype=np.uint8) * 30
            cv2.putText(self.image, "No images found", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(self.image, "Add images to the 'images' folder", (80, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def load_images(self):
        IMAGES_DIR.mkdir(exist_ok=True)
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        for ext in extensions:
            self.images.extend(sorted(IMAGES_DIR.glob(ext)))
        print(f"Loaded {len(self.images)} images from {IMAGES_DIR}")

    def load_current(self):
        if 0 <= self.index < len(self.images):
            self.image = cv2.imread(str(self.images[self.index]))
            if self.image is None:
                self.image = np.ones((512, 512, 3), dtype=np.uint8) * 30
        self.reset_view()

    def reset_view(self):
        self.zoom = MIN_ZOOM
        self.pan = None
        self.annotations = []

    def next_image(self):
        if self.images:
            self.index = (self.index + 1) % len(self.images)
            self.load_current()

    def previous_image(self):
        if self.images:
            self.index = (self.index - 1) % len(self.images)
            self.load_current()

    def _fit_scale(self):
        h, w = self.image.shape[:2]
        vw, vh = self.viewport_size
        return min(vw / w, vh / h)

    def _pan_center(self):
        if self.pan is not None:
            return self.pan
        h, w = self.image.shape[:2]
        return (w / 2.0, h / 2.0)

    def zoom_by(self, factor, anchor=None):
        """Multiply zoom by factor (e.g. 1.035 for a smooth per-frame step
        while a gesture is held). If anchor (original-image coords) is
        given, the pan eases toward it — this is what makes pinch-zoom feel
        like it's zooming toward the cursor instead of a fixed corner."""
        self.zoom = float(np.clip(self.zoom * factor, MIN_ZOOM, MAX_ZOOM))
        if self.zoom <= MIN_ZOOM:
            self.pan = None
        elif anchor is not None:
            self.pan_toward(anchor, ease=0.12)

    def pan_toward(self, target, ease=0.2):
        """Ease the pan center toward an original-image-space point."""
        if self.zoom <= MIN_ZOOM:
            return
        cx, cy = self._pan_center()
        tx, ty = target
        self.pan = (cx + (tx - cx) * ease, cy + (ty - cy) * ease)

    def adjust_brightness(self, delta):
        self.brightness = int(np.clip(self.brightness + delta, -100, 100))

    def adjust_contrast(self, factor):
        self.contrast = float(np.clip(self.contrast * factor, 0.4, 3.0))

    def toggle_invert(self):
        self.inverted = not self.inverted

    def update_cursor_position(self, cursor_pos, frame_shape):
        if cursor_pos is None:
            self.cursor_on_image = None
            return
        h_img, w_img = self.image.shape[:2]
        h_frame, w_frame = frame_shape
        img_x = int(cursor_pos[0] * w_img / w_frame)
        img_y = int(cursor_pos[1] * h_img / h_frame)
        img_x = max(0, min(img_x, w_img - 1))
        img_y = max(0, min(img_y, h_img - 1))
        self.cursor_on_image = (img_x, img_y)

    def add_annotation(self):
        if self.cursor_on_image:
            self.annotations.append(self.cursor_on_image)

    def undo_annotation(self):
        if self.annotations:
            self.annotations.pop()

    def clear_annotations(self):
        self.annotations = []

    def _crop_region(self):
        """(x1, y1, crop_w, crop_h, scale) describing the currently visible
        region of the original image and the scale to fill the viewport."""
        h, w = self.image.shape[:2]
        vw, vh = self.viewport_size
        scale = self._fit_scale() * self.zoom
        crop_w = min(vw / scale, w)
        crop_h = min(vh / scale, h)
        cx, cy = self._pan_center()
        x1 = float(np.clip(cx - crop_w / 2, 0, max(0, w - crop_w)))
        y1 = float(np.clip(cy - crop_h / 2, 0, max(0, h - crop_h)))
        return x1, y1, crop_w, crop_h, scale

    def show(self, base_override=None):
        """Render the current view: color/invert adjustments, crop+zoom to
        the fixed viewport size, then annotations/cursor drawn in viewport
        space so they stay correctly placed regardless of zoom/pan.

        base_override lets the caller substitute a same-size derived image
        (e.g. an AI heatmap overlay) as the base image.
        """
        source = base_override if base_override is not None else self.image
        vw, vh = self.viewport_size
        if source is None:
            return np.zeros((vh, vw, 3), dtype=np.uint8)

        img = source.copy()
        if self.brightness or self.contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=self.contrast, beta=self.brightness)
        if self.inverted:
            img = cv2.bitwise_not(img)

        x1, y1, crop_w, crop_h, scale = self._crop_region()
        x1i, y1i = int(round(x1)), int(round(y1))
        x2i = min(img.shape[1], x1i + max(1, int(round(crop_w))))
        y2i = min(img.shape[0], y1i + max(1, int(round(crop_h))))
        cropped = img[y1i:y2i, x1i:x2i]
        if cropped.size == 0:
            cropped = img
        display = cv2.resize(cropped, (vw, vh), interpolation=cv2.INTER_LINEAR)

        def to_viewport(pt):
            return (int(round((pt[0] - x1i) * scale)), int(round((pt[1] - y1i) * scale)))

        for i, point in enumerate(self.annotations):
            color = MARKER_COLORS[i % len(MARKER_COLORS)]
            vp = to_viewport(point)
            cv2.circle(display, vp, 6, color, -1)
            cv2.circle(display, vp, 10, color, 2)
            cv2.putText(display, f"P{i + 1}", (vp[0] + 12, vp[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if i > 0:
                prev_vp = to_viewport(self.annotations[i - 1])
                cv2.line(display, prev_vp, vp, color, 2)
                dist_px = np.hypot(point[0] - self.annotations[i - 1][0],
                                    point[1] - self.annotations[i - 1][1])
                mid = ((prev_vp[0] + vp[0]) // 2, (prev_vp[1] + vp[1]) // 2)
                text = f"{dist_px:.0f}px"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display, (mid[0] - tw // 2 - 5, mid[1] - th - 5),
                              (mid[0] + tw // 2 + 5, mid[1] + 5), (0, 0, 0), -1)
                cv2.putText(display, text, (mid[0] - tw // 2, mid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if self.cursor_on_image:
            cx, cy = to_viewport(self.cursor_on_image)
            if -30 <= cx <= vw + 30 and -30 <= cy <= vh + 30:
                cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
                cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)
                cv2.circle(display, (cx, cy), 10, (0, 255, 255), 1)
                cv2.circle(display, (cx, cy), 3, (0, 255, 255), -1)

        return display
