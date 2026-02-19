#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3 inference wrapper for mixed analysis.
Lazy-loads SAM3 modules to keep startup light.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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
        self._torch = None

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
        self._torch = torch
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

    def _maybe_empty_mps_cache(self) -> None:
        """MPS 캐시 해제 (메모리 누적 완화)."""
        try:
            if self._torch is None:
                return
            if str(self.device).lower() != "mps":
                return
            if hasattr(self._torch, "mps") and hasattr(self._torch.mps, "empty_cache"):
                self._torch.mps.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _to_numpy(arr):
        if hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    @staticmethod
    def _prepare_image_for_model(image: np.ndarray):
        image_for_model = image
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image_for_model = np.clip(image, 0, 255).astype(np.uint8)
            image_for_model = Image.fromarray(image_for_model)
        return image_for_model

    def _collect_segments_from_outputs(
        self,
        outputs: Dict[str, object],
        image_shape: Tuple[int, int],
        score_threshold: float,
        max_instances: int,
    ) -> List[Dict[str, object]]:
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
        h, w = image_shape
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
        self._maybe_empty_mps_cache()

        text_prompt = (prompt or "").strip() or "leaf"
        score_threshold = float(np.clip(score_threshold, 0.0, 1.0))
        max_instances = int(max(1, max_instances))

        try:
            image_for_model = self._prepare_image_for_model(image)
            state = self._processor.set_image(image_for_model)
            outputs = self._processor.set_text_prompt(text_prompt, state)
        finally:
            self._maybe_empty_mps_cache()

        h, w = image.shape[:2]
        return self._collect_segments_from_outputs(
            outputs=outputs,
            image_shape=(h, w),
            score_threshold=score_threshold,
            max_instances=max_instances,
        )

    def segment_image_with_points(
        self,
        image: np.ndarray,
        prompt: str,
        points: Sequence[Tuple[float, float]],
        point_labels: Sequence[int],
        score_threshold: float = 0.4,
        max_instances: int = 100,
    ) -> List[Dict[str, object]]:
        """SAM3 텍스트+포인트 프롬프트 기반 인스턴스 마스크 반환.

        Args:
            image: RGB image (H, W, 3)
            prompt: text prompt
            points: pixel-space point coordinates [(x, y), ...]
            point_labels: each point label (1=positive, 0=negative)
        """
        if image is None:
            return []

        if len(points) == 0 or len(points) != len(point_labels):
            return []

        self._ensure_ready()
        self._maybe_empty_mps_cache()
        torch = self._torch

        text_prompt = (prompt or "").strip() or "leaf"
        score_threshold = float(np.clip(score_threshold, 0.0, 1.0))
        max_instances = int(max(1, max_instances))

        try:
            image_for_model = self._prepare_image_for_model(image)
            state = self._processor.set_image(image_for_model)

            # Sam3Processor.set_text_prompt는 즉시 forward를 수행하므로
            # point prompting 경로에서는 텍스트 임베딩만 주입해 메모리 피크를 줄인다.
            with torch.inference_mode():
                text_outputs = self._model.backbone.forward_text([text_prompt], device=self.device)
            state["backbone_out"].update(text_outputs)
            if "geometric_prompt" not in state or state["geometric_prompt"] is None:
                state["geometric_prompt"] = self._model._get_dummy_prompt()

            h, w = image.shape[:2]
            den_x = float(max(1, w - 1))
            den_y = float(max(1, h - 1))
            normalized = []
            labels = []
            for (x, y), label in zip(points, point_labels):
                nx = float(np.clip(float(x) / den_x, 0.0, 1.0))
                ny = float(np.clip(float(y) / den_y, 0.0, 1.0))
                normalized.append((nx, ny))
                labels.append(1 if int(label) > 0 else 0)

            points_tensor = torch.tensor(
                normalized, dtype=torch.float32, device=self.device
            ).view(len(normalized), 1, 2)
            labels_tensor = torch.tensor(
                labels, dtype=torch.long, device=self.device
            ).view(len(labels), 1)
            state["geometric_prompt"].append_points(points_tensor, labels_tensor)
            outputs = self._processor._forward_grounding(state)
        finally:
            self._maybe_empty_mps_cache()

        return self._collect_segments_from_outputs(
            outputs=outputs,
            image_shape=(h, w),
            score_threshold=score_threshold,
            max_instances=max_instances,
        )
