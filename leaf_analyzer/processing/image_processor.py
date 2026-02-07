#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Processor for Advanced Leaf Analyzer
이미지 처리 및 마스크 생성 로직
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage.morphology import remove_small_holes, remove_small_objects

# Maxflow import
try:
    import maxflow as pymaxflow
except Exception:
    pymaxflow = None

import networkx as nx

# ΔE2000 함수 (폴백 포함)
try:
    from ..utils.color_analysis_utils import DeltaE2000 as _DE2000
    def delta_e2000(a, b):
        return _DE2000.delta_e_2000(a, b)
except Exception:
    # 폴백: 간단한 CIE76
    def delta_e2000(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.sqrt(np.sum((a - b) ** 2)))


class ImageProcessor:
    """이미지 처리 믹스인 클래스 - AdvancedLeafAnalyzer에서 상속받아 사용"""
    
    def _cached_resize_mask(self, mask: np.ndarray, target_size: tuple, mask_id: str = None) -> np.ndarray:
        """최적화: 캐싱을 사용한 마스크 리사이징
        
        Args:
            mask: 리사이징할 마스크
            target_size: (width, height) 목표 크기
            mask_id: 캐시 키로 사용할 고유 ID (None이면 자동 생성)
        """
        if mask is None or mask.size == 0:
            return np.zeros(target_size[::-1], dtype=np.uint8)
        
        # 캐시 키 생성 (mask 내용 기반 해시 + 버전)
        h, w = mask.shape[:2]
        step = max(1, h * w // 1000)  # 최대 1000개 샘플
        mask_flat = mask.flat[::step]
        mask_hash = hash(mask_flat.tobytes())
        base_id = f"{mask_hash}_{h}_{w}"
        if mask_id is None:
            mask_id = base_id
        else:
            mask_id = f"{mask_id}_{base_id}"
        
        cache_key = (mask_id, target_size, self._cache_version)
        
        # 캐시 히트
        if cache_key in self._resize_cache:
            return self._resize_cache[cache_key]
        
        # 캐시 미스: 리사이즈 수행
        resized = cv2.resize(
            mask.astype(np.uint8),
            target_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        # 캐시 크기 제한 (LRU 방식)
        if len(self._resize_cache) >= self._cache_max_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self._resize_cache))
            del self._resize_cache[oldest_key]
        
        self._resize_cache[cache_key] = resized
        return resized
    
    def _get_morph_kernel(self, size: int, shape=cv2.MORPH_ELLIPSE) -> np.ndarray:
        """최적화: 캐싱된 Morphology 커널 반환"""
        cache_key = (size, shape)
        if cache_key not in self._morph_kernels:
            self._morph_kernels[cache_key] = cv2.getStructuringElement(shape, (size, size))
        return self._morph_kernels[cache_key]
    
    def _batch_delta_e2000(self, colors1: np.ndarray, colors2: np.ndarray) -> np.ndarray:
        """최적화: 벡터화된 ΔE2000 계산 (CIE76 근사)
        
        Args:
            colors1: (N, 3) LAB 색상 배열
            colors2: (N, 3) LAB 색상 배열
        
        Returns:
            (N,) 거리 배열
        """
        colors1 = np.asarray(colors1, dtype=np.float32)
        colors2 = np.asarray(colors2, dtype=np.float32)
        return np.sqrt(np.sum((colors1 - colors2) ** 2, axis=-1))
    
    def _vectorized_min_distance(self, means: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        """최적화: 각 mean에서 가장 가까운 prototype까지의 거리 계산
        
        Args:
            means: (n_segments, 3) 세그먼트 평균 색상
            prototypes: (n_protos, 3) 프로토타입 색상
        
        Returns:
            (n_segments,) 최소 거리 배열
        """
        means = np.asarray(means, dtype=np.float32)
        prototypes = np.asarray(prototypes, dtype=np.float32)
        
        if prototypes.ndim == 1:
            prototypes = prototypes.reshape(1, -1)
        
        # Broadcasting: (n_segments, 1, 3) - (1, n_protos, 3)
        distances = np.sqrt(np.sum(
            (means[:, None, :] - prototypes[None, :, :]) ** 2,
            axis=2
        ))
        return distances.min(axis=1)
