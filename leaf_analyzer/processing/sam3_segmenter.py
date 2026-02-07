#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 inference wrapper for mixed analysis.
Lazy-loads SAM3 modules to keep startup light.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import os
import cv2
from PIL import Image


class Sam3Segmenter:
    """SAM3 이미지 추론 래퍼 (lazy load)."""

    def __init__(self, device: Optional[str] = None, compile_model: bool = False):
        self.device = device
        self.compile_model = compile_model
        self._model = None
        self._processor = None
        self._ready = False

    def _lazy_import(self):
        try:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:
            raise RuntimeError(
                "SAM3 모듈을 찾을 수 없습니다. 다음을 확인하세요:\n"
                "1) https://github.com/facebookresearch/sam3 설치\n"
                "2) HuggingFace 토큰 인증(hf auth login)\n"
                "3) 현재 환경에 sam3 패키지가 설치되어 있는지"
            ) from exc
        return torch, build_sam3_image_model, Sam3Processor

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        torch, build_sam3_image_model, Sam3Processor = self._lazy_import()
        requested = self.device.lower() if isinstance(self.device, str) else None
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if requested in (None, "", "mps"):
            if mps_available:
                self.device = "mps"
            else:
                if requested == "mps":
                    self.device = "cpu"
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested
        if self.device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self._model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            compile=self.compile_model,
        )
        self._processor = Sam3Processor(self._model, device=self.device)
        self._ready = True

    @staticmethod
    def _to_numpy(arr):
        if hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def segment_image(
        self,
        image: np.ndarray,
        prompt: str,
        score_threshold: float = 0.4,
        max_instances: int = 100,
    ) -> List[Dict[str, object]]:
        """SAM3 텍스트 프롬프트 기반 인스턴스 마스크 반환."""
        if image is None:
            return []

        self._ensure_ready()

        text_prompt = (prompt or "").strip() or "leaf"
        score_threshold = float(np.clip(score_threshold, 0.0, 1.0))
        max_instances = int(max(1, max_instances))

        image_for_model = image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image_for_model = np.clip(image, 0, 255).astype(np.uint8)
            image_for_model = Image.fromarray(image_for_model)
        state = self._processor.set_image(image_for_model)
        outputs = self._processor.set_text_prompt(text_prompt, state)

        masks = outputs.get("masks", None)
        scores = outputs.get("scores", None)
        if masks is None or scores is None:
            return []

        masks_np = self._to_numpy(masks)
        scores_np = self._to_numpy(scores).astype(np.float32)

        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0, :, :]
        elif masks_np.ndim != 3:
            raise RuntimeError(f"SAM3 마스크 차원 오류: {masks_np.shape}")

        count = min(masks_np.shape[0], scores_np.shape[0])
        order = np.argsort(scores_np[:count])[::-1]

        segments: List[Dict[str, object]] = []
        h, w = image.shape[:2]
        for idx in order:
            score = float(scores_np[idx])
            if score < score_threshold:
                continue
            mask = masks_np[idx] > 0.5
            if mask.shape[:2] != (h, w):
                resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                mask = resized > 0
            segments.append({"mask": mask, "score": score})
            if len(segments) >= max_instances:
                break

        return segments
