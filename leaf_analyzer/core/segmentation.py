#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GrabCut Segmenter for Advanced Leaf Analyzer
GrabCut 기반 전경/배경 분리기
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


class GrabCutSegmenter:
    """GrabCut 기반 전경/배경 분리기"""
    def __init__(self, iterations: int = 5, seed_radius: int = 4, rect_padding: int = 10):
        self.iterations = iterations
        self.seed_radius = seed_radius
        self.rect_padding = rect_padding

    def _compute_rect_from_seeds(self, seeds: List[Tuple[int, int]], image_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        if not seeds:
            return None
        ys = [y for x, y in seeds]
        xs = [x for x, y in seeds]
        min_x = max(min(xs) - self.rect_padding, 0)
        min_y = max(min(ys) - self.rect_padding, 0)
        max_x = min(max(xs) + self.rect_padding, image_shape[1] - 1)
        max_y = min(max(ys) + self.rect_padding, image_shape[0] - 1)
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        return (min_x, min_y, w, h)

    def _initialize_mask(self, image_shape: Tuple[int, int, int], fg_seeds: List[Tuple[int, int]], bg_seeds: List[Tuple[int, int]], rect: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        mask = np.full(image_shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        # 배경 시드: 확실한 배경
        for x, y in bg_seeds:
            cv2.circle(mask, (int(x), int(y)), self.seed_radius, cv2.GC_BGD, thickness=-1)
        # 전경 시드: 확실한 전경
        for x, y in fg_seeds:
            cv2.circle(mask, (int(x), int(y)), self.seed_radius, cv2.GC_FGD, thickness=-1)
        # ROI는 기본 PR_BGD 유지, 전경 시드 주변만 얇게 PR_FGD로 확장
        if rect is not None:
            x, y, w, h = rect
            pr_band = np.zeros(mask.shape, np.uint8)
            for px, py in fg_seeds:
                cv2.circle(pr_band, (int(px), int(py)), self.seed_radius + 3, 255, -1)
            band_roi = (pr_band[y:y+h, x:x+w] > 0)
            roi = mask[y:y+h, x:x+w]
            roi[np.logical_and(band_roi, roi == cv2.GC_PR_BGD)] = cv2.GC_PR_FGD
            mask[y:y+h, x:x+w] = roi
        # 이미지 외곽을 강한 배경으로 고정하여 경계 확산 방지
        h_img, w_img = image_shape[:2]
        ratio = float(getattr(self, 'settings', {}).get('grabcut_border_ratio', 0.005))
        border = max(1, int(ratio * min(h_img, w_img)))
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD
        return mask

    def segment(self, image_rgb: np.ndarray, fg_seeds: List[Tuple[int, int]], bg_seeds: List[Tuple[int, int]]) -> np.ndarray:
        if image_rgb is None or image_rgb.size == 0:
            return np.zeros((0, 0), dtype=bool)
            
        # 시드 검증: 전경 시드가 없으면 빈 마스크 반환
        if len(fg_seeds) == 0:
            print("전경 시드가 없어 GrabCut 실행 불가")
            return np.zeros(image_rgb.shape[:2], dtype=bool)
            
        rect = self._compute_rect_from_seeds(fg_seeds, image_rgb.shape)
        if rect is None:
            print("유효한 ROI를 계산할 수 없어 GrabCut 실행 불가")
            return np.zeros(image_rgb.shape[:2], dtype=bool)
            
        mask = self._initialize_mask(image_rgb.shape, fg_seeds, bg_seeds, rect)
        
        # 마스크에 전경/배경 샘플이 있는지 확인
        fg_pixels = np.sum((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
        bg_pixels = np.sum((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD))
        
        if fg_pixels == 0 or bg_pixels == 0:
            print(f"GrabCut 샘플 부족: fg={fg_pixels}, bg={bg_pixels}")
            # 간단한 색상 기반 폴백
            return self._color_based_fallback(image_rgb, fg_seeds)
            
        try:
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(image_rgb, mask, rect, bgdModel, fgdModel, self.iterations, cv2.GC_INIT_WITH_MASK)
            result = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
            return result.astype(bool)
        except cv2.error as e:
            print(f"GrabCut 실행 실패: {e}")
            return self._color_based_fallback(image_rgb, fg_seeds)
    
    def _color_based_fallback(self, image_rgb: np.ndarray, fg_seeds: List[Tuple[int, int]]) -> np.ndarray:
        """GrabCut 실패 시 색상 기반 폴백"""
        if len(fg_seeds) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=bool)
            
        # 시드 주변 색상 샘플링
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        
        for x, y in fg_seeds:
            if 0 <= x < w and 0 <= y < h:
                # 시드 주변 50x50 영역의 색상 범위 기반 검출
                roi_size = 25
                x1, y1 = max(0, x-roi_size), max(0, y-roi_size)
                x2, y2 = min(w, x+roi_size), min(h, y+roi_size)
                
                # 시드 색상
                seed_color = image_rgb[y, x]
                
                # 색상 거리 기반 마스크
                roi = image_rgb[y1:y2, x1:x2]
                color_dist = np.linalg.norm(roi - seed_color, axis=2)
                local_mask = color_dist < 30  # 임계값
                
                # 전체 마스크에 추가
                mask[y1:y2, x1:x2] |= local_mask
        
        return mask
