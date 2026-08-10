"""Real chest X-ray pathology detection.

Uses torchxrayvision's pretrained DenseNet121 (trained on the NIH
ChestX-ray14 dataset) for real pathology scores, plus a Grad-CAM heatmap
computed from the model's real gradients — no brightness heuristics, no
fabricated heatmap shapes.

A caveat worth being upfront about: these scores are calibrated per-pathology
against an operating threshold (`model.op_threshs`), not a single shared
probability scale, so most classes cluster near 0.5 on almost any input.
Comparing scores *within one image* (which pathology stands out relative to
the others) is far more meaningful than reading any single score as "chance
of disease." `analyze()` ranks findings by that relative standout (a z-score
across the image's own pathology scores) rather than by raw score alone.
This is a research-grade demo model, not a diagnostic device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
import torchxrayvision as xrv

PATHOLOGY_INFO = {
    "Atelectasis": ("Partial or complete collapse of lung tissue", "Lower lobes"),
    "Consolidation": ("Alveolar air replaced by fluid or solid material", "Lung parenchyma"),
    "Infiltration": ("Substance denser than air within lung parenchyma", "Diffuse"),
    "Pneumothorax": ("Air in the pleural space, visible pleural line", "Pleural space"),
    "Edema": ("Fluid accumulation in the lung interstitium", "Perihilar"),
    "Emphysema": ("Hyperinflated lungs with flattened diaphragm", "Upper lobes"),
    "Fibrosis": ("Scarring with a reticular or reticulonodular pattern", "Lung bases"),
    "Effusion": ("Fluid in the pleural space, blunting costophrenic angles", "Costophrenic angles"),
    "Pneumonia": ("Infection causing alveolar inflammation", "Lung lobes"),
    "Pleural_Thickening": ("Thickened pleural membrane with scarring", "Pleural surface"),
    "Cardiomegaly": ("Enlarged cardiac silhouette", "Mediastinum"),
    "Nodule": ("Small focal opacity under 3cm in diameter", "Variable"),
    "Mass": ("Solid lesion over 3cm in diameter", "Variable"),
    "Hernia": ("Protrusion of abdominal contents through the diaphragm", "Diaphragm"),
    "Lung Lesion": ("Focal abnormal lung tissue", "Variable"),
    "Fracture": ("Break in bone continuity", "Ribs/thorax"),
    "Lung Opacity": ("Region of increased density in the lung field", "Variable"),
    "Enlarged Cardiomediastinum": ("Widened mediastinal silhouette", "Mediastinum"),
}

MIN_PROBABILITY = 0.4   # floor: ignore findings the model itself scores low in absolute terms
Z_SCORE_THRESHOLD = 1.0  # how far above this image's own baseline a finding must stand out


@dataclass
class Finding:
    condition: str
    probability: float
    z_score: float
    description: str
    region: str


@dataclass
class AnalysisResult:
    is_normal: bool
    findings: list
    heatmap_overlay: Optional[np.ndarray]
    error: Optional[str] = None

    @property
    def primary(self) -> Optional[Finding]:
        return self.findings[0] if self.findings else None


class AIEngine:
    """Loads the model once; call analyze() per image."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading chest X-ray AI model (device={self.device}) ...")
        self.model = xrv.models.DenseNet(weights="densenet121-res224-nih")
        self.model = self.model.to(self.device).eval()

        self._transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ])

        self._activations = None
        self._cam_target = self._find_cam_target()
        if self._cam_target is not None:
            self._cam_target.register_forward_hook(self._forward_hook)

        num_pathologies = len([p for p in self.model.pathologies if p])
        print(f"AI model ready. {num_pathologies} pathology classes.")

    def _find_cam_target(self):
        """Locate the final conv feature block to hook for Grad-CAM."""
        if hasattr(self.model, "features"):
            return self.model.features
        return None

    def _forward_hook(self, module, inp, out):
        self._activations = out

    def _preprocess(self, bgr_image: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        normalized = xrv.datasets.normalize(gray, 255)
        normalized = normalized[None, ...]
        transformed = self._transform(normalized)
        tensor = torch.from_numpy(transformed.copy()).float().unsqueeze(0)
        return tensor.to(self.device)

    def _grad_cam(self, tensor: torch.Tensor, class_index: int, out_size) -> Optional[np.ndarray]:
        if self._cam_target is None:
            return None

        self._activations = None
        output = self.model(tensor)
        if self._activations is None:
            return None

        score = output[0, class_index]
        gradients = torch.autograd.grad(score, self._activations, retain_graph=False)[0]

        activations = self._activations[0].detach()
        weights = gradients[0].mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, out_size)

    def analyze(self, bgr_image: np.ndarray) -> AnalysisResult:
        if bgr_image is None or bgr_image.size == 0:
            return AnalysisResult(is_normal=True, findings=[], heatmap_overlay=None,
                                   error="No image")

        try:
            h, w = bgr_image.shape[:2]
            tensor = self._preprocess(bgr_image)

            with torch.no_grad():
                outputs = self.model(tensor)
            probs = outputs[0].cpu().numpy()
        except Exception as exc:
            return AnalysisResult(is_normal=True, findings=[], heatmap_overlay=None,
                                   error=f"{type(exc).__name__}: {exc}")

        names = [n for n in self.model.pathologies if n]
        values = np.array([p for n, p in zip(self.model.pathologies, probs) if n], dtype=np.float32)
        mean, std = float(values.mean()), float(values.std())
        z_scores = (values - mean) / std if std > 1e-6 else np.zeros_like(values)

        findings = []
        for name, prob, z in zip(names, values, z_scores):
            prob = float(prob)
            z = float(z)
            if prob < MIN_PROBABILITY or z < Z_SCORE_THRESHOLD:
                continue
            description, region = PATHOLOGY_INFO.get(name, ("", ""))
            findings.append(Finding(condition=name, probability=prob, z_score=z,
                                     description=description, region=region))

        findings.sort(key=lambda f: f.z_score, reverse=True)
        is_normal = not findings

        # Always compute a heatmap for whichever pathology scores highest,
        # even below the "flagged finding" bar — otherwise toggling the
        # heatmap on a normal-looking image does nothing visible.
        heatmap_overlay = None
        try:
            top_name = names[int(np.argmax(z_scores))]
            pathologies = list(self.model.pathologies)
            top_index = pathologies.index(top_name)
            cam = self._grad_cam(tensor, top_index, (w, h))
            if cam is not None:
                heatmap_color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_overlay = cv2.addWeighted(bgr_image, 0.6, heatmap_color, 0.4, 0)
        except Exception:
            heatmap_overlay = None  # classification is still valid without a heatmap

        return AnalysisResult(is_normal=is_normal, findings=findings,
                               heatmap_overlay=heatmap_overlay)
