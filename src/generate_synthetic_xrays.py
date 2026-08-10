"""Cartoon chest X-ray shapes for UI/gesture testing only.

These are drawn with OpenCV primitives (ellipses, lines) — they are not real
anatomy and the AI engine's predictions on them are not meaningful. Useful
only for exercising the gesture/zoom/viewer UI without a webcam pointed at
real images. For an actually meaningful AI demo, use fetch_sample_images.py
instead, which downloads real openly-licensed radiographs.

Output goes to images/synthetic/ (not images/) so it never overwrites the
real sample set used by default.
"""
import numpy as np
import cv2
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "images" / "synthetic"


def create_realistic_xray(filename, pattern='normal'):
    """Draw a cartoon chest X-ray shape (not real anatomy)."""
    img = np.ones((1024, 1024), dtype=np.uint8) * 30
    
    # Ribcage
    cv2.ellipse(img, (512, 600), (350, 450), 0, 0, 360, 180, -1)
    
    # Left lung
    left_lung = img.copy()
    cv2.ellipse(left_lung, (380, 500), (180, 300), 0, 0, 360, 40, -1)
    
    # Right lung
    right_lung = img.copy()
    cv2.ellipse(right_lung, (644, 500), (180, 300), 0, 0, 360, 40, -1)
    
    img = cv2.addWeighted(img, 0.7, left_lung, 0.3, 0)
    img = cv2.addWeighted(img, 0.7, right_lung, 0.3, 0)
    
    # Heart
    cv2.ellipse(img, (480, 550), (120, 150), -10, 0, 360, 220, -1)
    
    # Diaphragm
    cv2.line(img, (100, 750), (400, 650), 180, 3)
    cv2.line(img, (600, 650), (900, 750), 180, 3)
    
    # Ribs
    for i in range(6):
        y = 350 + i * 60
        cv2.ellipse(img, (512, y), (350, 15), 0, 0, 180, 140, 1)
    
    # Spine
    cv2.rectangle(img, (500, 200), (524, 800), 150, -1)
    
    if pattern == 'pneumonia':
        for _ in range(20):
            cx = np.random.randint(300, 750)
            cy = np.random.randint(500, 750)
            radius = np.random.randint(10, 40)
            cv2.circle(img, (cx, cy), radius, np.random.randint(180, 230), -1)
        cv2.ellipse(img, (350, 650), (100, 120), 0, 0, 360, 210, -1)
    
    elif pattern == 'cardiomegaly':
        cv2.ellipse(img, (480, 550), (160, 180), -10, 0, 360, 235, -1)
    
    elif pattern == 'effusion':
        cv2.rectangle(img, (100, 700), (400, 800), 200, -1)
        cv2.rectangle(img, (600, 700), (900, 800), 200, -1)
    
    elif pattern == 'pneumothorax':
        cv2.rectangle(img, (150, 100), (400, 350), 20, -1)
        cv2.line(img, (150, 350), (400, 200), 200, 2)
    
    elif pattern == 'mass':
        cv2.circle(img, (700, 350), 70, 200, -1)
        cv2.circle(img, (700, 350), 75, 220, 2)
    
    elif pattern == 'nodule':
        cv2.circle(img, (500, 450), 25, 190, -1)
        cv2.circle(img, (500, 450), 28, 210, 1)
    
    # Noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Blur
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    
    cv2.imwrite(str(OUTPUT_DIR / filename), img)
    print(f'Created: {filename}')


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patterns = [
        ('normal_1.jpg', 'normal'),
        ('normal_2.jpg', 'normal'),
        ('pneumonia_left.jpg', 'pneumonia'),
        ('pneumonia_bilateral.jpg', 'pneumonia'),
        ('cardiomegaly.jpg', 'cardiomegaly'),
        ('pleural_effusion.jpg', 'effusion'),
        ('pneumothorax.jpg', 'pneumothorax'),
        ('lung_mass.jpg', 'mass'),
        ('nodule.jpg', 'nodule'),
    ]

    print('Generating cartoon chest X-ray shapes for UI testing (NOT real anatomy)...\n')
    for filename, pattern in patterns:
        create_realistic_xray(filename, pattern)

    print(f'\nDone! Created {len(patterns)} synthetic images in {OUTPUT_DIR}')
    print('These are for UI/gesture testing only — not used by the app by default.')