#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphological Analyzer for Advanced Leaf Analyzer
형태학적 분석기
"""

import cv2
import numpy as np


class MorphologicalAnalyzer:
    """형태학적 분석기"""
    
    @staticmethod
    def analyze_contour(contour):
        """단일 윤곽선 분석"""
        # 면적
        area = cv2.contourArea(contour)
        
        # 둘레
        perimeter = cv2.arcLength(contour, True)
        
        # 회전된 경계 상자 (OBB)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        
        # 길이, 너비, 회전 각도
        center, (width, height), angle = rect
        length = max(width, height)
        width = min(width, height)
        
        # 기타 지표
        aspect_ratio = length / width if width > 0 else 0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        return {
            "area": area,
            "perimeter": perimeter,
            "length": length,
            "width": width,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "angle": angle,
            "center": center,
            "bounding_box": box
        }

    @staticmethod
    def analyze_mask_with_holes(binary_mask: np.ndarray):
        """이진 마스크에서 홀(내부 구멍)을 반영하여 지표 계산
        - 면적: 외곽(+), 홀(−) 면적의 부호 합 또는 mask.sum()
        - 둘레: 외곽+홀 경계선 둘레 합
        - 길이/너비/각도/OBB: 외곽(가장 큰) 컨투어 기준
        - contour: 외곽(가장 큰) 컨투어 반환
        """
        mask_u8 = binary_mask.astype(np.uint8)
        area_px = int(np.sum(mask_u8 > 0))
        contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "area": 0.0,
                "perimeter": 0.0,
                "length": 0.0,
                "width": 0.0,
                "aspect_ratio": 0.0,
                "circularity": 0.0,
                "angle": 0.0,
                "center": (0.0, 0.0),
                "bounding_box": np.zeros((4,2), dtype=np.float32),
                "contour": None
            }
        # 둘레: 외곽+홀 모두 합산
        perimeter_sum = 0.0
        for c in contours:
            perimeter_sum += cv2.arcLength(c, True)
        # 외곽 컨투어(부모=-1) 중 최대 면적 컨투어 선택
        ext_indices = []
        if hierarchy is not None:
            h = hierarchy[0]
            for i, info in enumerate(h):
                parent = int(info[3])
                if parent == -1:
                    ext_indices.append(i)
        else:
            ext_indices = list(range(len(contours)))
        if not ext_indices:
            ext_indices = list(range(len(contours)))
        ext_areas = [(i, cv2.contourArea(contours[i])) for i in ext_indices]
        main_idx = max(ext_areas, key=lambda t: t[1])[0] if ext_areas else 0
        main_contour = contours[main_idx]
        # 길이/너비/각도/OBB/중심 (타원 지향 OBB 시도 → 실패 시 minAreaRect)
        pts = np.asarray(main_contour, dtype=np.float32).reshape(-1, 2)
        H, W = mask_u8.shape[:2]
        def _near_border(pxy: np.ndarray, margin: int = 2) -> bool:
            if pxy.size == 0:
                return False
            xs, ys = pxy[:, 0], pxy[:, 1]
            return (xs.min() <= margin or xs.max() >= W - 1 - margin or
                    ys.min() <= margin or ys.max() >= H - 1 - margin)
        use_ellipse = False
        quad = None
        ang_deg = 0.0
        if pts.shape[0] >= 20 and not _near_border(pts, margin=max(2, int(0.01 * min(H, W)))):
            try:
                if pts.shape[0] >= 5:
                    (cx, cy), (ax0, ax1), ang = cv2.fitEllipse(pts)
                    # 주/부축 정렬 및 각도 보정(major axis 기준)
                    a_major = max(ax0, ax1)
                    a_minor = min(ax0, ax1)
                    theta = np.deg2rad(ang)
                    if ax0 < ax1:
                        theta += np.pi / 2.0
                    # 이심률로 길쭉함 평가
                    if a_major > 1e-6:
                        ecc = float(np.sqrt(max(0.0, 1.0 - (a_minor / a_major) ** 2)))
                    else:
                        ecc = 0.0
                    # 타원방향 사용 조건: 충분히 길쭉함
                    if ecc >= 0.6:
                        c = np.array([cx, cy], dtype=np.float32)
                        u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                        v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
                        # 중심 기준 투영
                        P = pts - c
                        t_u = P @ u
                        t_v = P @ v
                        # 극단치 완화: 퍼센타일 클리핑(1%~99%)
                        lo_u, hi_u = np.percentile(t_u, [1.0, 99.0])
                        lo_v, hi_v = np.percentile(t_v, [1.0, 99.0])
                        corners_uv = np.array([
                            [lo_u, lo_v], [hi_u, lo_v], [hi_u, hi_v], [lo_u, hi_v]
                        ], dtype=np.float32)
                        quad = (corners_uv[:, 0:1] * u + corners_uv[:, 1:2] * v) + c
                        ang_deg = float(np.rad2deg(theta))
                        use_ellipse = True
            except Exception:
                use_ellipse = False
                quad = None
        if not use_ellipse:
            rect = cv2.minAreaRect(main_contour)
            quad = cv2.boxPoints(rect)
            center, (w_rect, h_rect), ang_deg = rect
        else:
            center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        # 길이/너비 계산 (quad 변 길이)
        d = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        side1, side2 = float(d[0]), float(d[1])
        length = max(side1, side2)
        width = min(side1, side2)
        aspect_ratio = (length / width) if width > 0 else 0.0
        circularity = (4.0 * np.pi * area_px) / (perimeter_sum ** 2) if perimeter_sum > 0 else 0.0
        return {
            "area": float(area_px),
            "perimeter": float(perimeter_sum),
            "length": float(length),
            "width": float(width),
            "aspect_ratio": float(aspect_ratio),
            "circularity": float(circularity),
            "angle": float(ang_deg),
            "center": (float(center[0]), float(center[1])),
            "bounding_box": quad,
            "contour": main_contour
        }
    
    @staticmethod
    def analyze_image_objects(binary_mask):
        """이미지 내 모든 객체 분석"""
        # 연결 성분 찾기
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        objects = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 100:  # 최소 면적 필터
                obj_data = MorphologicalAnalyzer.analyze_contour(contour)
                obj_data["id"] = i
                obj_data["contour"] = contour
                objects.append(obj_data)
        
        return objects
