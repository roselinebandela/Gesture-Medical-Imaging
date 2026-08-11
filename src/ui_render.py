from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_FILES = {
    "light": ["C:/Windows/Fonts/segoeuil.ttf", "C:/Windows/Fonts/segoeui.ttf"],
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"],
    "semibold": ["C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf",
                 "C:/Windows/Fonts/arial.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arial.ttf"],
}
_FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_font_cache = {}


def _load_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key in _font_cache:
        return _font_cache[key]
    for path in _FONT_FILES.get(weight, _FONT_FILES["regular"]) + [_FALLBACK]:
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            _font_cache[key] = font
            return font
    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def put_text(img: np.ndarray, text: str, pos, size: int = 18,
             color=(255, 255, 255), weight: str = "regular") -> np.ndarray:
    """Draw anti-aliased text onto a BGR OpenCV image, returns the image."""
    font = _load_font(size, weight)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_text_multi(img: np.ndarray, spans, pos, size: int = 18, weight: str = "regular"):
    """Draw several (text, color) spans left-to-right starting at pos in one call."""
    font = _load_font(size, weight)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    x, y = pos
    for text, color in spans:
        rgb_color = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=rgb_color)
        x += draw.textlength(text, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def text_size(text: str, size: int = 18, weight: str = "regular"):
    font = _load_font(size, weight)
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_panel(img, x, y, w, h, color, alpha=0.15):
    """Semi-transparent filled rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def draw_rounded_panel(img, x, y, w, h, color, alpha=0.55, radius=14):
    """Semi-transparent filled rectangle with rounded corners."""
    overlay = img.copy()
    r = min(radius, w // 2, h // 2)
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def draw_rounded_rect(img, x, y, w, h, color, thickness=1, radius=10):
    """Rounded-rectangle outline (or filled if thickness<0)."""
    r = min(radius, w // 2, h // 2)
    pts_top = ((x + r, y), (x + w - r, y))
    pts_bottom = ((x + r, y + h), (x + w - r, y + h))
    pts_left = ((x, y + r), (x, y + h - r))
    pts_right = ((x + w, y + r), (x + w, y + h - r))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
        return img
    cv2.line(img, *pts_top, color, thickness, cv2.LINE_AA)
    cv2.line(img, *pts_bottom, color, thickness, cv2.LINE_AA)
    cv2.line(img, *pts_left, color, thickness, cv2.LINE_AA)
    cv2.line(img, *pts_right, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    return img


def draw_palm_icon(img, center, radius, color, thickness=2):
    cx, cy = center
    cv2.circle(img, (cx, cy + radius // 2), radius, color, thickness)
    for i in range(5):
        angle = np.pi * (0.15 + i * 0.175)
        x2 = int(cx + np.cos(np.pi - angle) * radius * 1.6)
        y2 = int(cy + radius // 2 - np.sin(angle) * radius * 1.6)
        cv2.line(img, (cx, cy), (x2, y2), color, thickness, cv2.LINE_AA)
    return img


def draw_fist_icon(img, center, radius, color, thickness=2):
    cv2.circle(img, center, radius, color, thickness)
    return img


def draw_pinch_icon(img, center, radius, color, thickness=2):
    cx, cy = center
    cv2.circle(img, (cx - radius // 2, cy), radius // 3, color, thickness)
    cv2.circle(img, (cx + radius // 2, cy), radius // 3, color, thickness)
    return img


def draw_arrow_icon(img, center, size, direction, color, thickness=2):
    cx, cy = center
    if direction == "left":
        pts = [(cx + size, cy - size), (cx - size, cy), (cx + size, cy + size)]
    else:
        pts = [(cx - size, cy - size), (cx + size, cy), (cx - size, cy + size)]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
    return img


def draw_dot(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)
    return img
