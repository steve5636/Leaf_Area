#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Utilities for Advanced Leaf Analyzer
Export를 위한 유틸리티 함수들 (OBB, Polygon, RLE 등)
"""

import cv2
import numpy as np
from typing import List


class ExportUtils:
    """Export 유틸리티 믹스인 클래스"""

    def _bbox_from_mask(self, mask: np.ndarray):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        return x_min, y_min, x_max, y_max


    def _rle_counts_from_mask(self, mask: np.ndarray) -> List[int]:
        """COCO Uncompressed RLE 생성 (column-major, 0-run 시작)
        mask: HxW, bool/uint8
        반환: counts 리스트 (0/1 런 길이)
        """
        m = (mask.astype(np.uint8) > 0)
        # column-major로 1차원화 (Fortran order)
        flat = m.flatten(order='F').astype(np.uint8)
        if flat.size == 0:
            return [0]
        # 값 변화 지점 인덱스 계산 → 연속 구간 길이로 변환
        diff_idx = np.where(flat[1:] != flat[:-1])[0] + 1
        run_lengths = np.diff(np.r_[0, diff_idx, flat.size]).tolist()
        # COCO 규칙: 항상 0-run부터 시작
        if flat[0] == 1:
            run_lengths = [0] + run_lengths
        return run_lengths
    

    def _mask_to_polygon(self, mask: np.ndarray, epsilon_factor: float = 0.001) -> List[List[float]]:
        """마스크를 COCO polygon 형식으로 변환 (내부 홀 포함)
        mask: HxW, bool/uint8
        epsilon_factor: contour 단순화 계수 (둘레 대비 비율)
        반환: [[x1, y1, x2, y2, ...], [...]] (외곽선 + 내부 홀 polygon)
        """
        m = (mask.astype(np.uint8) > 0).astype(np.uint8)
        # RETR_CCOMP: 외곽선과 내부 홀을 계층적으로 검출
        contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            return [[]]
        
        polygons = []
        for i, contour in enumerate(contours):
            # 너무 작은 contour는 무시 (최소 3개 점 필요)
            if contour.shape[0] < 3:
                continue
            
            # Contour 단순화 (점 개수 줄이기)
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 단순화 후에도 최소 3개 점이 있어야 유효한 polygon
            if approx.shape[0] < 3:
                continue
            
            # COCO 형식: [x1, y1, x2, y2, ...] (flatten)
            polygon = approx.flatten().tolist()
            
            # 최소 6개 값 (3개 점) 필요
            if len(polygon) >= 6:
                polygons.append(polygon)
        
        return polygons if polygons else [[]]


    def _yolo_bbox_line(self, class_id: int, x_min: int, y_min: int, x_max: int, y_max: int, img_w: int, img_h: int) -> str:
        # YOLO xywh normalized
        cx = (x_min + x_max) / 2.0 / img_w
        cy = (y_min + y_max) / 2.0 / img_h
        w = (x_max - x_min) / float(img_w)
        h = (y_max - y_min) / float(img_h)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


    def _order_corners_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """pts: (4,2) in pixels. Return CW order starting from top-left."""
        pts = np.asarray(pts, dtype=float).reshape(4, 2)
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:,1] - c[1], pts[:,0] - c[0])
        order_ccw = np.argsort(ang)  # CCW
        pts = pts[order_ccw]
        # start from top-left: min y, then min x
        start = np.lexsort((pts[:,0], pts[:,1]))[0]
        pts = np.roll(pts, -start, axis=0)
        # force clockwise: if CCW, reverse order except first
        if np.cross(pts[1]-pts[0], pts[2]-pts[1]) > 0:  # CCW
            pts = pts[[0,3,2,1]]
        return pts


    def _yolo_obb_line(self, class_id: int, corners_xy: np.ndarray, img_w: int, img_h: int) -> str:
        # corners_xy: (4,2) in pixels → normalize with epsilon clamp
        pts = self._order_corners_clockwise(corners_xy.astype(float))
        eps = 1e-6
        norm = []
        for (x, y) in pts:
            nx = min(1.0 - eps, max(0.0, float(x) / float(img_w)))
            ny = min(1.0 - eps, max(0.0, float(y) / float(img_h)))
            norm.extend([nx, ny])
        return f"{class_id} " + " ".join(f"{v:.6f}" for v in norm)

    # --- Robust OBB building utilities ---

    def _touching_border(self, pts: np.ndarray, img_w: int, img_h: int, margin: int | None = None) -> bool:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if pts.size == 0:
            return False
        if margin is None:
            # 이미지 크기 대비 동적 마진 (1% 이상, 최소 2px)
            margin = max(2, int(0.01 * float(min(img_w, img_h))))
        xs, ys = pts[:, 0], pts[:, 1]
        return (xs.min() <= margin or xs.max() >= img_w - 1 - margin or
                ys.min() <= margin or ys.max() >= img_h - 1 - margin)


    def _convex_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            return pts
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        return hull


    def _pca_obb(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        c = pts.mean(axis=0, keepdims=True)
        X = pts - c
        if X.shape[0] < 2:
            return np.tile(c, (4, 1))
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        R = Vt  # principal axes
        proj = X @ R.T
        mn, mx = proj.min(axis=0), proj.max(axis=0)
        box_proj = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
        return (box_proj @ R) + c


    def _minarea_obb(self, pts: np.ndarray) -> np.ndarray:
        rect = cv2.minAreaRect(np.asarray(pts, dtype=np.float32))
        return cv2.boxPoints(rect)


    def _aabb(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)

    def _scale_obb_from_mask(self, smask: np.ndarray, img_w: int, img_h: int) -> np.ndarray | None:
        """Scale 전용: SAM3 폴리곤에 맞춰 사각형 OBB 생성."""
        try:
            m = (smask.astype(np.uint8) > 0).astype(np.uint8)
            if m.size == 0 or np.sum(m) == 0:
                return None
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            cnt = max(cnts, key=cv2.contourArea)
            if cnt is None or len(cnt) < 4:
                return None
            peri = cv2.arcLength(cnt, True)
            eps = max(2.0, 0.02 * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) == 4:
                quad = approx.reshape(-1, 2).astype(float)
                if self._touching_border(quad, img_w, img_h):
                    return self._aabb(quad)
                return quad
            # 폴리곤이 4점이 아니면 minAreaRect 폴백
            pts = cnt.reshape(-1, 2).astype(np.float32)
            quad = self._minarea_obb(pts)
            return quad.astype(float)
        except Exception:
            return None


    def _reasonable_box(self, quad: np.ndarray) -> bool:
        try:
            if quad is None or len(quad) != 4:
                return False
            area = cv2.contourArea(quad.astype(np.float32))
            if area < 1:
                return False
            d = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            ar = (d.max() + 1e-6) / (d.min() + 1e-6)
            return ar < 25
        except Exception:
            return False


    def _robust_obb_from_points(self, pts: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) < 10:
            return self._aabb(pts)
        hull = self._convex_points(pts)
        # 0) 경계 접촉이면 회전 금지: AABB로 강제
        if self._touching_border(pts, img_w, img_h):
            return self._aabb(hull).astype(float)
        # 1) 타원 지향 OBB 시도 (길쭉한 Leaf 안정화)
        quad = None
        try:
            H, W = img_h, img_w
            if hull.shape[0] >= 20:
                h32 = hull.astype(np.float32)
                (cx, cy), (ax0, ax1), ang = cv2.fitEllipse(h32)
                a_major = max(ax0, ax1)
                a_minor = min(ax0, ax1)
                theta = np.deg2rad(ang)
                if ax0 < ax1:
                    theta += np.pi / 2.0
                ecc = float(np.sqrt(max(0.0, 1.0 - (a_minor / max(a_major, 1e-6)) ** 2))) if a_major > 1e-6 else 0.0
                if ecc >= 0.6:  # 충분히 길쭉할 때만 사용
                    c = np.array([cx, cy], dtype=np.float32)
                    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
                    P = h32 - c
                    t_u = P @ u
                    t_v = P @ v
                    lo_u, hi_u = np.percentile(t_u, [1.0, 99.0])
                    lo_v, hi_v = np.percentile(t_v, [1.0, 99.0])
                    corners_uv = np.array([[lo_u, lo_v], [hi_u, lo_v], [hi_u, hi_v], [lo_u, hi_v]], dtype=np.float32)
                    quad = (corners_uv[:, 0:1] * u + corners_uv[:, 1:2] * v) + c
        except Exception:
            quad = None
        # 2) 기본은 minAreaRect (타원 실패/부적합 시)
        if quad is None or not self._reasonable_box(quad):
            quad = self._minarea_obb(hull)
        # 3) 품질 저하 시 PCA-OBB로 보정
        if not self._reasonable_box(quad):
            quad = self._pca_obb(hull)
        # 4) 여전히 이상하면 AABB
        if not self._reasonable_box(quad):
            quad = self._aabb(hull)
        return quad.astype(float)


