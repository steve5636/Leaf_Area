#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask Generator for Advanced Leaf Analyzer
마스크 생성 및 처리 로직
"""

import cv2
import numpy as np
from typing import List, Tuple
from skimage.morphology import remove_small_holes, remove_small_objects

from ..core.segmentation import GrabCutSegmenter


class MaskGenerator:
    """마스크 생성 믹스인 클래스"""

    def _apply_convex_hull_to_scale_mask(self, scale_mask: np.ndarray) -> np.ndarray:
        """Scale 마스크에 Convex Hull 적용하여 내부 hole 제거
        
        Args:
            scale_mask: 원본 Scale 마스크 (bool 또는 uint8)
            
        Returns:
            np.ndarray: Convex Hull이 적용된 마스크 (bool 또는 uint8)
        """
        if scale_mask is None or np.sum(scale_mask) == 0:
            return scale_mask
        
        from scipy import ndimage
        
        # 원본 타입 저장
        original_dtype = scale_mask.dtype
        is_bool = original_dtype == bool
        
        # uint8로 변환
        mask_uint8 = scale_mask.astype(np.uint8)
        
        # 연결 성분별로 Convex Hull 적용
        scale_labels, num_scales = ndimage.label(mask_uint8)
        
        if num_scales == 0:
            return scale_mask
        
        # 새 마스크 생성
        h, w = scale_mask.shape
        convex_mask = np.zeros((h, w), dtype=np.uint8)
        
        for scale_id in range(1, num_scales + 1):
            # 각 Scale 객체 추출
            single_scale = (scale_labels == scale_id).astype(np.uint8)
            
            # 외곽선 찾기
            contours, _ = cv2.findContours(single_scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            # 가장 큰 외곽선 선택
            main_contour = max(contours, key=cv2.contourArea)
            
            # Convex Hull 계산
            hull = cv2.convexHull(main_contour)
            
            # Convex Hull을 마스크에 그리기
            cv2.fillConvexPoly(convex_mask, hull, 1)
        
        print(f"   → Scale Convex Hull 적용: {np.sum(scale_mask)}픽셀 → {np.sum(convex_mask)}픽셀")
        
        # 원본 타입으로 복원
        if is_bool:
            return convex_mask.astype(bool)
        else:
            return convex_mask

    def _preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """세그멘테이션용 전처리 (색상 분리 강화)"""
        if image is None:
            return image
        if not self.settings.get("preprocess_enabled", True):
            return image
        method = str(self.settings.get("preprocess_method", "bilateral")).lower()
        if method == "none":
            return image
        try:
            if method == "meanshift":
                sp = int(self.settings.get("pre_meanshift_sp", 10))
                sr = int(self.settings.get("pre_meanshift_sr", 20))
                return cv2.pyrMeanShiftFiltering(image, sp=sp, sr=sr)
            # 기본: bilateral
            d = int(self.settings.get("pre_bilateral_d", 5))
            sigma_color = float(self.settings.get("pre_bilateral_sigma_color", 50))
            sigma_space = float(self.settings.get("pre_bilateral_sigma_space", 50))
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        except Exception:
            # 전처리 실패 시 원본 반환
            return image
    

    def _gather_background_seeds_for_leaf(self) -> List[Tuple[int, int]]:
        """Leaf 분할용 BG 시드 모음 (배경 + 스케일)"""
        bg_seeds = list(self.seed_manager.seeds.get("background", []))
        scale_seeds = list(self.seed_manager.seeds.get("scale", []))
        if scale_seeds:
            bg_seeds.extend(scale_seeds)
        # 중복 제거
        if not bg_seeds:
            return []
        return list({(int(x), int(y)) for x, y in bg_seeds})
    

    def generate_leaf_mask(self) -> np.ndarray:
        """GrabCut으로 Leaf 마스크 생성"""
        return self.generate_grabcut_mask()
    

    def generate_grabcut_mask(self) -> np.ndarray:
        """현재 시드들로 GrabCut 마스크 생성"""
        if self.original_image is None:
            return np.zeros((0, 0), dtype=bool)
        
        leaf_seeds = self.seed_manager.seeds.get("leaf", [])
        bg_seeds = self._gather_background_seeds_for_leaf()
        
        print("   → OpenCV GrabCut 사용 (BG 시드 포함)")
        print(f"   → Leaf seed {len(leaf_seeds)}개, BG seed {len(bg_seeds)}개")
        
        # 전처리 적용
        base_img = self._preprocess_for_segmentation(self.original_image)
        
        # GrabCut 반복 횟수 증가로 정확도 향상
        self.grabcut.iterations = 10  # 기본 5 → 10
        mask = self.grabcut.segment(base_img, leaf_seeds, bg_seeds)
        return self._postprocess_leaf_mask(mask)
    

    def _postprocess_leaf_mask(self, mask: np.ndarray) -> np.ndarray:
        """공통 후처리 파이프라인 (얇은 가지 보존 포함)"""
        if mask is None or mask.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        
        mask = mask.astype(np.uint8)
        
        # 1. Bilateral filter로 에지 보존 평활화
        d = int(self.settings.get("bilateral_d", 9))
        sigma_color = float(self.settings.get("bilateral_sigma_color", 75))
        sigma_space = float(self.settings.get("bilateral_sigma_space", 75))
        mask_filtered = cv2.bilateralFilter(mask * 255, d, sigma_color, sigma_space)
        mask = (mask_filtered > 127).astype(np.uint8)
        
        # 2. 적응형 형태학적 연산 (얇은 가지 보존)
        kernel_size = int(self.settings.get("morphology_kernel_size", 5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        use_thin_kernel = False
        if np.any(mask):
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            median_dist = float(np.median(dist[mask > 0]))
            thin_thresh = float(self.settings.get("thin_branch_dist_thresh", 2.4))
            if median_dist > 0 and median_dist < thin_thresh:
                use_thin_kernel = True
        if use_thin_kernel:
            thin_scale = float(self.settings.get("thin_branch_kernel_scale", 0.6))
            k = max(3, int(round(kernel_size * thin_scale)))
            if k % 2 == 0:
                k += 1
            kernel = self._get_morph_kernel(k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            kernel = self._get_morph_kernel(kernel_size)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. 크기 기반 필터링
        try:
            from skimage.morphology import remove_small_holes, remove_small_objects
            # 구멍 메우기: 동적 임계
            hole_ratio = float(self.settings.get("hole_ratio_max", 0.025))
            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            if num_labels > 1:
                unique_labels, label_counts = np.unique(labels, return_counts=True)
                valid_mask = unique_labels > 0
                unique_labels = unique_labels[valid_mask]
                label_counts = label_counts[valid_mask]
                for label_id, component_area in zip(unique_labels, label_counts):
                    if component_area > 0:
                        component_mask = (labels == label_id)
                        hole_thresh = int(hole_ratio * component_area)
                        filled = remove_small_holes(component_mask, area_threshold=max(50, hole_thresh))
                        mask[component_mask] = filled[component_mask].astype(np.uint8)
            
            # 작은 객체 제거 (시드 포함 컴포넌트 보존)
            small_ratio = float(self.settings.get("small_obj_ratio", 0.8))
            min_obj = max(100, int(small_ratio * int(self.settings.get("min_object_area", 3000))))
            mask_bool = mask.astype(bool)
            keep_seed_mask = None
            leaf_seeds = self.seed_manager.seeds.get("leaf", [])
            if leaf_seeds and np.any(mask_bool):
                _, labels2 = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
                seed_labels = set()
                h, w = labels2.shape
                for sx, sy in leaf_seeds:
                    if 0 <= sy < h and 0 <= sx < w:
                        lbl = int(labels2[sy, sx])
                        if lbl > 0:
                            seed_labels.add(lbl)
                if seed_labels:
                    keep_seed_mask = np.isin(labels2, np.array(list(seed_labels), dtype=np.int32))
            cleaned = remove_small_objects(mask_bool, min_size=min_obj)
            if keep_seed_mask is not None:
                cleaned = np.logical_or(cleaned, keep_seed_mask)
            mask = cleaned.astype(np.uint8)
        except Exception as e:
            print(f"형태학적 후처리 오류: {e}")
        
        # 4. 경계 처리 개선
        if self.settings.get("remove_border_touches", False):
            h, w = mask.shape
            num, labs = cv2.connectedComponents(mask, connectivity=8)
            if num > 1:
                border_width = 3
                border_mask = np.zeros_like(mask, dtype=bool)
                border_mask[:border_width, :] = True
                border_mask[-border_width:, :] = True
                border_mask[:, :border_width] = True
                border_mask[:, -border_width:] = True
                
                unique_labels, label_counts = np.unique(labs, return_counts=True)
                valid_mask = unique_labels > 0
                unique_labels = unique_labels[valid_mask]
                label_counts = label_counts[valid_mask]
                for label_id, total_area in zip(unique_labels, label_counts):
                    if total_area > 0:
                        component_mask = (labs == label_id)
                        border_area = np.sum(component_mask & border_mask)
                        if (border_area / total_area) > 0.5:
                            mask[component_mask] = 0
        
        # 5. 시드 복구
        mask_bool = mask.astype(bool)
        mask_recovered = self._recover_missed_seeds(mask_bool)
        return mask_recovered


    def _recover_missed_seeds(self, mask: np.ndarray) -> np.ndarray:
        """전역 분할에서 누락된 FG 시드를 로컬 GrabCut으로 복구"""
        img = self.original_image
        h, w = img.shape[:2]
        recovered = mask.copy().astype(bool)
        
        # 시드 밀도에 따라 동적 윈도우 크기 결정
        leaf_seeds = self.seed_manager.seeds.get("leaf", [])
        if len(leaf_seeds) == 0:
            return recovered.astype(np.uint8)
            
        # 시드 밀도 계산
        seed_density = len(leaf_seeds) / (h * w) * 1000000  # 시드/메가픽셀
        
        # 밀도가 높으면 작은 윈도우, 낮으면 큰 윈도우
        if seed_density > 50:  # 고밀도
            R = int(np.clip(0.05 * min(h, w), 40, 100))
            coverage_thresh = 0.2
        elif seed_density > 20:  # 중간 밀도
            R = int(np.clip(0.08 * min(h, w), 50, 140))
            coverage_thresh = 0.25
        else:  # 저밀도
            R = int(np.clip(0.12 * min(h, w), 60, 180))
            coverage_thresh = 0.3
            
        for (sx, sy) in leaf_seeds:
            sx = int(np.clip(sx, 0, w-1))
            sy = int(np.clip(sy, 0, h-1))
            
            # 커버리지 체크
            y0, y1 = max(0, sy - R), min(h-1, sy + R)
            x0, x1 = max(0, sx - R), min(w-1, sx + R)
            local = recovered[y0:y1+1, x0:x1+1]
            fg_ratio = float(local.sum()) / float(local.size + 1e-6)
            
            # 시드가 커버되지 않았거나 커버리지가 부족한 경우
            if not recovered[sy, sx] or fg_ratio < coverage_thresh:
                # 로컬 GrabCut 수행
                x0, y0 = max(0, sx - R), max(0, sy - R)
                x1, y1 = min(w-1, sx + R), min(h-1, sy + R)
                rect = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
                
                # 초기 마스크 설정 (동적 반경)
                init = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
                
                # 시드 중심으로 확실한 전경
                fg_radius = int(0.25 * R) if seed_density > 30 else int(0.30 * R)
                cv2.circle(init, (sx, sy), fg_radius, cv2.GC_FGD, -1)
            
                # 가능한 전경 영역
                pr_radius = int(0.50 * R) if seed_density > 30 else int(0.55 * R)
                cv2.circle(init, (sx, sy), pr_radius, cv2.GC_PR_FGD, -1)
            
                # 경계 배경 설정
                border = 5
                init[:y0+border, :] = cv2.GC_BGD
                init[y1-border+1:, :] = cv2.GC_BGD
                init[:, :x0+border] = cv2.GC_BGD
                init[:, x1-border+1:] = cv2.GC_BGD
            
                # GrabCut 수행
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                local = init.copy()
                try:
                    cv2.grabCut(img, local, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                    local_fg = (local == cv2.GC_FGD) | (local == cv2.GC_PR_FGD)
                    recovered |= local_fg
                except Exception:
                    pass  # GrabCut 실패 시 무시
                    
        return recovered.astype(np.uint8)

    def _generate_scale_mask_from_seeds(self, scale_seeds: List[Tuple[int, int]]) -> np.ndarray:
        """일반 GrabCut 모드에서도 Scale seed로 독립적인 GrabCut 실행"""
        if self.original_image is None or len(scale_seeds) == 0:
            return np.zeros((0, 0), dtype=bool)
            
        print(f"   → GrabCut 모드: Scale seed {len(scale_seeds)}개로 독립적인 GrabCut 실행")
        
        # Scale seed만으로 독립적인 GrabCut 실행
        try:
            scale_grabcut = GrabCutSegmenter(iterations=8)  # Scale용 독립적 GrabCut
            scale_mask = scale_grabcut.segment(self.original_image, scale_seeds, [])
            
            total_scale_pixels = np.sum(scale_mask)
            print(f"   → 독립 GrabCut Scale 픽셀: {total_scale_pixels}개")
            
            if total_scale_pixels > 0:
                # 형태학적 후처리 (최적화: 커널 캐싱)
                kernel = self._get_morph_kernel(3)
                scale_mask = cv2.morphologyEx(scale_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                scale_mask = cv2.morphologyEx(scale_mask, cv2.MORPH_OPEN, kernel)
                
                # 최소 면적 필터링 적용 (Leaf와 동일한 기준)
                scale_mask = self._apply_size_filter(scale_mask.astype(bool))
                
                # Convex Hull 적용하여 내부 hole 제거
                scale_mask = self._apply_convex_hull_to_scale_mask(scale_mask)
                
                final_pixels = np.sum(scale_mask)
                print(f"   → 후처리 + 면적필터 + Convex Hull 후 독립 Scale 픽셀: {final_pixels}개")
            
            return scale_mask.astype(bool)
            
        except Exception as e:
            print(f"   ⚠️ Scale 독립 GrabCut 실패: {e}, 색상 기반으로 폴백")
            # 색상 기반 폴백
            h, w = self.original_image.shape[:2]
            scale_mask = np.zeros((h, w), dtype=bool)
            
            for i, (sx, sy) in enumerate(scale_seeds):
                sx, sy = int(np.clip(sx, 0, w-1)), int(np.clip(sy, 0, h-1))
                seed_color = self.original_image[sy, sx].astype(np.float32)
                
                roi_size = 50
                x1, y1 = max(0, sx - roi_size//2), max(0, sy - roi_size//2)
                x2, y2 = min(w, sx + roi_size//2), min(h, sy + roi_size//2)
                
                roi = self.original_image[y1:y2, x1:x2]
                color_diff = np.linalg.norm(roi - seed_color, axis=2)
                local_mask = color_diff < 30.0
                scale_mask[y1:y2, x1:x2] |= local_mask
            
            # 색상 기반 폴백에도 최소 면적 필터링 + Convex Hull 적용
            if np.sum(scale_mask) > 0:
                scale_mask = self._apply_size_filter(scale_mask.astype(bool))
                scale_mask = self._apply_convex_hull_to_scale_mask(scale_mask)
                print(f"   → 색상 폴백 + 면적필터 + Convex Hull 후 Scale 픽셀: {np.sum(scale_mask)}개")
            
            return scale_mask.astype(bool)
    

    def _generate_scale_mask(self) -> np.ndarray:
        """스케일 마스크 생성 (Scale seed 기반 GrabCut 사용)"""
        if self.original_image is None:
            return np.zeros((0, 0), dtype=bool)
            
        scale_seeds = self.seed_manager.seeds.get("scale", [])
        print(f"   → Scale seed 개수: {len(scale_seeds)}개")
        
        if len(scale_seeds) == 0:
            # Scale seed가 없으면 빈 마스크 반환 (검출 안함)
            print("   → Scale seed 없음, Scale 검출 스킵")
            return np.zeros(self.original_image.shape[:2], dtype=bool)
        
        img = self.working_image if hasattr(self, 'working_image') and self.working_image is not None else self.original_image
        h, w = img.shape[:2]
        
        # Scale seed별로 개별 GrabCut 적용
        scale_mask = np.zeros((h, w), dtype=bool)
        
        for i, (sx, sy) in enumerate(scale_seeds):
            sx, sy = int(np.clip(sx, 0, w-1)), int(np.clip(sy, 0, h-1))
            print(f"   → Scale seed {i+1}: ({sx}, {sy})")
            
            # 시드 주변 영역을 ROI로 설정 (더 큰 영역)
            roi_size = 150  # 100 → 150으로 증가
            x1, y1 = max(0, sx - roi_size//2), max(0, sy - roi_size//2)
            x2, y2 = min(w, sx + roi_size//2), min(h, sy + roi_size//2)
            
            # 로컬 GrabCut 적용
            roi_img = img[y1:y2, x1:x2]
            roi_h, roi_w = roi_img.shape[:2]
            
            if roi_h > 10 and roi_w > 10:  # ROI가 충분히 큰 경우만
                # 로컬 좌표로 변환
                local_sx, local_sy = sx - x1, sy - y1
                
                # 로컬 GrabCut 설정
                rect = (5, 5, roi_w-10, roi_h-10)  # 전체 ROI를 범위로
                mask = np.full((roi_h, roi_w), cv2.GC_PR_BGD, np.uint8)
                
                # 시드 주변을 확실한 전경으로 설정
                cv2.circle(mask, (local_sx, local_sy), 8, cv2.GC_FGD, -1)
                cv2.circle(mask, (local_sx, local_sy), 15, cv2.GC_PR_FGD, -1)
                
                # ROI 경계를 배경으로 설정
                mask[:3, :] = cv2.GC_BGD
                mask[-3:, :] = cv2.GC_BGD
                mask[:, :3] = cv2.GC_BGD
                mask[:, -3:] = cv2.GC_BGD
                
                try:
                    # GrabCut 실행
                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)
                    cv2.grabCut(roi_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                    
                    # 결과 추출
                    local_scale_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
                    local_pixels = np.sum(local_scale_mask)
                    
                    print(f"     → 로컬 GrabCut 결과: {local_pixels}픽셀 검출")
                    
                    # 전체 마스크에 병합
                    scale_mask[y1:y2, x1:x2] |= local_scale_mask
                    
                except Exception as e:
                    print(f"     → 로컬 GrabCut 실패: {e}, 색상 기반으로 폴백")
                    
                    # 색상 기반 폴백
                    roi_r = roi_img[:, :, 0].astype(np.float32)
                    roi_g = roi_img[:, :, 1].astype(np.float32)
                    roi_b = roi_img[:, :, 2].astype(np.float32)
                    
                    # 완화된 Scale 검출 조건
                    minR = max(80, self.easy_params["minR"] * 0.6)  # 더 완화된 조건
                    ratR = max(1.1, self.easy_params["ratR"] * 0.8)  # 더 완화된 비율
                    
                    local_scale_mask = (
                        (roi_r > minR) & 
                        (roi_g * ratR < roi_r) & 
                        (roi_b * ratR < roi_r)
                    )
                    
                    color_pixels = np.sum(local_scale_mask)
                    print(f"     → 색상 기반 폴백: {color_pixels}픽셀 검출")
                    
                    # 전체 마스크에 병합
                    scale_mask[y1:y2, x1:x2] |= local_scale_mask
        
        # 형태학적 후처리
        total_scale_pixels = np.sum(scale_mask)
        print(f"   → 총 Scale 픽셀: {total_scale_pixels}개")
        
        if total_scale_pixels > 0:
            kernel = self._get_morph_kernel(3)
            scale_mask = cv2.morphologyEx(scale_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            scale_mask = cv2.morphologyEx(scale_mask, cv2.MORPH_OPEN, kernel)
            
            final_pixels = np.sum(scale_mask)
            print(f"   → 후처리 후 Scale 픽셀: {final_pixels}개")
        
        return scale_mask.astype(bool)
    

    def _detect_scale_by_color(self) -> np.ndarray:
        """색상 기반 Scale 검출 (seed 없을 때)"""
        if self.original_image is None:
            return np.zeros((0, 0), dtype=bool)
            
        img = self.working_image if hasattr(self, 'working_image') and self.working_image is not None else self.original_image
        h, w = img.shape[:2]
        
        print(f"   → 색상 기반 Scale 검출 시도 (이미지: {w}x{h})")
        
        # 빠른 처리를 위해 이미지 축소
        scale_factor = 4 if max(h, w) > 2000 else 2
        small_img = cv2.resize(img, (w//scale_factor, h//scale_factor))
        
        # RGB 채널 분리
        r_channel = small_img[:, :, 0].astype(np.float32)
        g_channel = small_img[:, :, 1].astype(np.float32)
        b_channel = small_img[:, :, 2].astype(np.float32)
        
        # 완화된 Scale 검출 조건 (다양한 스케일 객체 대응)
        results = []
        
        # 여러 파라미터 조합 시도
        param_sets = [
            {"minR": 120, "ratR": 1.3},  # 연한 빨간색
            {"minR": 150, "ratR": 1.5},  # 중간 빨간색
            {"minR": 180, "ratR": 1.8},  # 진한 빨간색
            {"minR": 200, "ratR": 2.0},  # 매우 진한 빨간색
        ]
        
        for j, params in enumerate(param_sets):
            minR, ratR = params["minR"], params["ratR"]
            
            scale_mask_temp = (
                (r_channel > minR) & 
                (g_channel * ratR < r_channel) & 
                (b_channel * ratR < r_channel)
            )
            
            pixels_found = np.sum(scale_mask_temp)
            results.append((pixels_found, scale_mask_temp))
            print(f"     → 파라미터 {j+1} (minR={minR}, ratR={ratR:.1f}): {pixels_found}픽셀")
        
        # 가장 많은 픽셀을 찾은 결과 선택
        best_result = max(results, key=lambda x: x[0])
        scale_mask = best_result[1]
        
        print(f"   → 최종 선택: {best_result[0]}픽셀")
        
        # 원본 크기로 확대
        scale_mask_full = cv2.resize(scale_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 형태학적 후처리 (최적화: 커널 캐싱)
        if np.sum(scale_mask_full) > 0:
            kernel = self._get_morph_kernel(5)
            scale_mask_full = cv2.morphologyEx(scale_mask_full, cv2.MORPH_CLOSE, kernel)
            scale_mask_full = cv2.morphologyEx(scale_mask_full, cv2.MORPH_OPEN, kernel)
            
            final_pixels = np.sum(scale_mask_full)
            print(f"   → 최종 Scale 픽셀: {final_pixels}개")
        
        return scale_mask_full.astype(bool)
    

    def _apply_size_filter(self, raw_mask: np.ndarray) -> np.ndarray:
        """원본 마스크에 최소 면적 필터링만 적용 (빠른 후처리)"""
        if raw_mask is None or raw_mask.size == 0:
            return np.zeros((0, 0), dtype=bool)
            
        try:
            # 현재 설정된 최소 면적
            min_area = self.manual_settings.get("min_area", 200)
            print(f"   → 최소 면적 필터링: {min_area}픽셀 이상")
            
            # 연결 성분 분석
            num_labels, labels = cv2.connectedComponents(raw_mask.astype(np.uint8), connectivity=8)
            if num_labels <= 1:
                return np.zeros_like(raw_mask, dtype=bool)
            # 라벨별 픽셀수 (라벨 0=배경 제외)
            flat = labels.reshape(-1)
            counts = np.bincount(flat, minlength=num_labels)
            keep_ids = np.where(counts >= int(min_area))[0]
            # 배경(0) 제거
            keep_ids = keep_ids[keep_ids != 0].astype(np.int32)
            if keep_ids.size == 0:
                print("   → 필터링 결과: 0개 객체 유지")
                return np.zeros_like(raw_mask, dtype=bool)
            # 벡터화된 라벨 포함 여부
            kept = np.isin(labels, keep_ids)
            print(f"   → 필터링 결과: {keep_ids.size}개 객체, {int(kept.sum())}픽셀 유지")
            return kept
            
        except Exception as e:
            print(f"   → 필터링 실패: {e}, 원본 마스크 반환")
            return raw_mask.astype(bool)
    

