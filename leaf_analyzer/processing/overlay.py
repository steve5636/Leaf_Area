#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay Manager for Advanced Leaf Analyzer
오버레이 및 시각화 생성
"""

import cv2
import numpy as np
from typing import Optional, Set, List, Union
from PIL import Image, ImageTk


class OverlayManager:
    """오버레이 생성 믹스인 클래스"""

    def _draw_count_label(self, image_rgb: np.ndarray, text: str) -> np.ndarray:
        """좌상단 카운트 텍스트 렌더링 (공통)"""
        return self._draw_text_block(image_rgb, [text])
    
    def _get_contour_thickness(self, extra: int = 1) -> int:
        """윤곽선 두께를 기본값보다 약간 두껍게 반환."""
        try:
            base = int(self.settings.get("overlay_contour_thickness", 1))
        except Exception:
            base = 1
        return int(max(1, base + int(extra)))

    def _build_rainbow_palette(self, num_colors: int = 72) -> List[tuple[int, int, int]]:
        """Leaf용 레인보우 팔레트 (RGB). 세션 단위로 랜덤 셔플."""
        n = max(8, int(num_colors))
        hsv = np.zeros((n, 1, 3), dtype=np.uint8)
        hsv[:, 0, 0] = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
        hsv[:, 0, 1] = 255
        hsv[:, 0, 2] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        palette = [tuple(int(v) for v in rgb[i, 0]) for i in range(n)]
        # 세션 단위 랜덤 셔플로 저개수에서도 다양한 색상 보장
        if not hasattr(self, "_overlay_palette_seed"):
            try:
                self._overlay_palette_seed = int(np.random.randint(0, 2**31 - 1))
            except Exception:
                self._overlay_palette_seed = 0
        rng = np.random.default_rng(self._overlay_palette_seed)
        order = rng.permutation(n)
        return [palette[int(i)] for i in order]

    def _draw_text_block(self, image_rgb: np.ndarray, lines: List[str], origin: tuple[int, int] = (10, 10)) -> np.ndarray:
        """좌상단 텍스트 블록 렌더링 (여러 줄 지원)"""
        try:
            overlay = image_rgb
            font = cv2.FONT_HERSHEY_SIMPLEX
            try:
                overlay_font_scale = float(self.settings.get("overlay_font_scale", 0.45))
                overlay_font_thickness = int(self.settings.get("overlay_font_thickness", 1))
            except Exception:
                overlay_font_scale, overlay_font_thickness = 0.45, 1
            
            if not lines:
                return overlay
            
            sizes = [cv2.getTextSize(str(line), font, overlay_font_scale, overlay_font_thickness)[0] for line in lines]
            max_w = max([s[0] for s in sizes]) if sizes else 0
            max_h = max([s[1] for s in sizes]) if sizes else 0
            line_h = max_h + 6
            x0, y0 = origin
            box_w = max_w + 10
            box_h = line_h * len(lines) + 4
            cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
            for i, line in enumerate(lines):
                y_line = y0 + 5 + max_h + i * line_h
                cv2.putText(overlay, str(line), (x0 + 5, y_line), font, overlay_font_scale, (255, 255, 255), overlay_font_thickness)
            return overlay
        except Exception:
            return image_rgb

    def _build_stats_from_masks(self, leaf_mask, scale_mask=None, leaf_area_thresh: int = 1, scale_area_thresh: int = 1) -> dict:
        """마스크 기반 stats 생성 (미리보기/분석 공통)"""
        try:
            num_leaves = 0
            num_scales = 0
            if leaf_mask is not None and np.sum(leaf_mask) > 0:
                l_contours, _ = cv2.findContours(leaf_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_leaves = len([c for c in l_contours if cv2.contourArea(c) >= leaf_area_thresh])
            if scale_mask is not None and np.sum(scale_mask) > 0:
                s_contours, _ = cv2.findContours(scale_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                num_scales = len([c for c in s_contours if cv2.contourArea(c) >= scale_area_thresh])
        except Exception:
            num_leaves, num_scales = 0, 0
        return {
            "active_leaf_count": int(num_leaves),
            "active_scale_count": int(num_scales),
            "deleted_leaf_count": int(len(getattr(self, '_deleted_objects', set()))),
            "deleted_scale_count": int(len(getattr(self, '_deleted_scale_objects', set()))),
        }
    
    def _overlay_instances(self, image_rgb: np.ndarray, binary_mask: np.ndarray, palette_type: str = 'leaf', alpha: float = 0.35, contour_thickness: int = 1) -> np.ndarray:
        """이진 마스크를 연결 성분 기준으로 다색 오버레이.
        - image_rgb: RGB 이미지(H,W,3)
        - binary_mask: HxW bool/uint8
        - palette_type: 'leaf' | 'scale'
        - alpha: 색상 블렌딩 비율
        - contour_thickness: 윤곽선 두께
        """
        try:
            img = image_rgb
            if binary_mask is None or binary_mask.size == 0:
                return img
            m = (binary_mask.astype(np.uint8) > 0).astype(np.uint8)
            h, w = m.shape[:2]
            if h == 0 or w == 0:
                return img
            num_labels, labels = cv2.connectedComponents(m, connectivity=8)
            if num_labels <= 1:
                return img
            # 팔레트 (RGB) - 시각적으로 구분 쉬운 고채도 색상들
            if palette_type == 'scale':
                # 고대비 36색 팔레트 (빨강/주황/분홍/청록/파랑/보라 계열)
                palette = [
                    (255,0,0),(255,100,0),(255,200,0),(255,0,100),(255,0,200),(255,100,100),(255,150,0),
                    (255,50,150),(255,0,50),(255,150,150),(255,80,80),(255,180,100),(200,0,0),(220,50,0),
                    (0,255,255),(0,220,255),(0,180,255),(0,255,180),(0,255,220),(100,255,255),(0,200,200),
                    (50,255,255),(0,150,255),(100,220,255),(80,255,200),(0,180,220),(0,140,200),(0,100,255),
                    (150,0,255),(200,0,255),(180,0,220),(255,0,255),(220,0,200),(200,100,255),(150,100,255),(180,80,255)
                ]
            else:
                # Leaf는 레인보우 팔레트로 더 구분성 높임
                palette = self._build_rainbow_palette(72)
            overlay = img.astype(np.float32)
            for lid in range(1, num_labels):
                comp = (labels == lid)
                if not np.any(comp):
                    continue
                color = np.array(palette[(lid - 1) % len(palette)], dtype=np.float32)
                overlay[comp] = overlay[comp] * (1.0 - alpha) + color * alpha
                try:
                    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        outline_color = (255, 255, 255)
                        if palette_type == 'leaf':
                            outline_color = tuple(int(v) for v in color)
                        cv2.drawContours(overlay, cnts, -1, outline_color, contour_thickness)
                except Exception:
                    pass
            return overlay.clip(0, 255).astype(np.uint8)
        except Exception:
            return image_rgb

    def _overlay_by_labels(self, image_rgb: np.ndarray, labels_map: np.ndarray, include_ids: Union[List[int], Set[int]], palette_type: str = 'leaf', alpha: float = 0.35, contour_thickness: int = 1, highlight_ids: Optional[Set[int]] = None) -> np.ndarray:
        """인스턴스 라벨맵(ID별)로 다색 오버레이. 같은 ID는 떨어져 있어도 같은 색.
        - image_rgb: HxWx3 RGB
        - labels_map: HxW int32 라벨맵 (0=배경)
        - include_ids: 렌더링할 ID 집합
        - palette_type: 'leaf'|'scale'
        - alpha: 블렌딩 비율
        - contour_thickness: 윤곽선 두께
        - highlight_ids: 굵은 윤곽 강조할 ID 집합(선택)
        """
        try:
            img = image_rgb.astype(np.float32)
            if labels_map is None or labels_map.size == 0:
                return image_rgb
            ids = [int(i) for i in np.unique(labels_map) if int(i) > 0]
            ids = [i for i in ids if i in include_ids]
            if not ids:
                return image_rgb
            if palette_type == 'scale':
                palette = [
                    (255,0,0),(255,100,0),(255,200,0),(255,0,100),(255,0,200),(255,100,100),(255,150,0),
                    (255,50,150),(255,0,50),(255,150,150),(255,80,80),(255,180,100),(200,0,0),(220,50,0),
                    (0,255,255),(0,220,255),(0,180,255),(0,255,180),(0,255,220),(100,255,255),(0,200,200),
                    (50,255,255),(0,150,255),(100,220,255),(80,255,200),(0,180,220),(0,140,200),(0,100,255),
                    (150,0,255),(200,0,255),(180,0,220),(255,0,255),(220,0,200),(200,100,255),(150,100,255),(180,80,255)
                ]
            else:
                palette = self._build_rainbow_palette(72)
            for idx, oid in enumerate(ids):
                comp = (labels_map == int(oid))
                if not np.any(comp):
                    continue
                color = np.array(palette[(hash(oid) % len(palette))], dtype=np.float32)
                img[comp] = img[comp] * (1.0 - alpha) + color * alpha
                try:
                    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    thick = contour_thickness
                    if highlight_ids is not None and int(oid) in highlight_ids:
                        thick = contour_thickness + 2
                    if cnts:
                        outline_color = (255, 255, 255)
                        if palette_type == 'leaf':
                            outline_color = tuple(int(v) for v in color)
                        cv2.drawContours(img, cnts, -1, outline_color, int(max(1, thick)))
                except Exception:
                    pass
            return img.clip(0, 255).astype(np.uint8)
        except Exception:
            return image_rgb

    def add_seed_markers(self, image):
        """시드 마커 추가"""
        colors = {
            "leaf": (0, 255, 0),
            "scale": (255, 0, 0),
            "background": (0, 128, 255)
        }
        
        for seed_class, color in colors.items():
            for x, y in self.seed_manager.seeds[seed_class]:
                # 원본 좌표를 표시 좌표로 변환
                display_x = int(x * self.display_scale)
                display_y = int(y * self.display_scale)
                
                # 마커 그리기
                cv2.circle(image, (display_x, display_y), 5, color, -1)
                cv2.circle(image, (display_x, display_y), 6, (255, 255, 255), 1)
        
        return image

    def create_preview_overlay(self, leaf_mask, scale_mask=None, stats: dict | None = None):
        """미리보기 오버레이 이미지 생성 (Scale 마스크 포함). 항상 상단 카운트/윤곽선/X표시 표시
        - 디스플레이 크기에서 직접 렌더링하여 결과 오버레이와 스타일 일치
        """
        # 디스플레이 이미지 보장
        if not hasattr(self, 'display_image') or self.display_image is None:
            self.update_display_image()
        base_img = self.display_image.copy() if hasattr(self, 'display_image') and self.display_image is not None else None
        
        if base_img is None:
            print("베이스 이미지가 None입니다!")
            return None
            
        # 디스플레이 크기에서 처리
        h, w = base_img.shape[:2]
        print(f"베이스 이미지 크기(디스플레이): {w}x{h}")
        
        try:
            overlay_font_scale = float(self.settings.get("overlay_font_scale", 0.45))
            overlay_font_thickness = int(self.settings.get("overlay_font_thickness", 1))
        except Exception:
            overlay_font_scale, overlay_font_thickness = 0.45, 1
        overlay_contour_thickness = self._get_contour_thickness()
    
        preview_img = base_img.copy()
        # 최적화: 캐싱된 리사이즈 사용
        preview_leaf_mask = self._cached_resize_mask(leaf_mask, (w, h), 'preview_leaf') > 0 if leaf_mask is not None and leaf_mask.size > 0 else None
        preview_scale_mask = self._cached_resize_mask(scale_mask, (w, h), 'preview_scale') > 0 if scale_mask is not None and scale_mask.size > 0 else None
            
        # 디버그: 마스크 상태 확인
        leaf_pixels = np.sum(preview_leaf_mask) if preview_leaf_mask is not None else 0
        scale_pixels = np.sum(preview_scale_mask) if preview_scale_mask is not None else 0
        print(f"적용할 마스크: leaf={leaf_pixels}픽셀, scale={scale_pixels}픽셀")
        
        # 잎 마스크 오버레이 (객체별 다양한 색상) + 윤곽선 - 투명도 낮춤
        if preview_leaf_mask is not None and np.sum(preview_leaf_mask) > 0:
            preview_img = self._overlay_instances(preview_img, preview_leaf_mask, palette_type='leaf', alpha=0.2, contour_thickness=overlay_contour_thickness)
            print("잎 마스크(다색) 오버레이 적용")
        
        # Scale 마스크 오버레이 (객체별 다양한 색상) + 윤곽선 - 투명도 낮춤
        if preview_scale_mask is not None and np.sum(preview_scale_mask) > 0:
            preview_img = self._overlay_instances(preview_img, preview_scale_mask, palette_type='scale', alpha=0.2, contour_thickness=overlay_contour_thickness)
            print("Scale 마스크(다색) 오버레이 적용")
        
        # 상단 카운트 표기 (Preview)
        if stats is None:
            stats = self._build_stats_from_masks(
                preview_leaf_mask.astype(np.uint8) if preview_leaf_mask is not None else None,
                preview_scale_mask.astype(np.uint8) if preview_scale_mask is not None else None,
                leaf_area_thresh=1,
                scale_area_thresh=1
            )
        num_leaves = int(stats.get("active_leaf_count", 0))
        num_scales = int(stats.get("active_scale_count", 0))
        lines = [f"Leaves: {num_leaves}, Scales: {num_scales} (Preview)"]
        del_leaf = int(stats.get("deleted_leaf_count", 0))
        del_scale = int(stats.get("deleted_scale_count", 0))
        if del_leaf > 0 or del_scale > 0:
            lines.append(f"Deleted: Leaf {del_leaf}, Scale {del_scale}")
        preview_img = self._draw_text_block(preview_img, lines)
    
        # 삭제된 객체 X 표시 (Leaf/Scale)
        try:
            # Leaf X 표시: 원본 라벨에서 중심 계산 후 디스플레이 스케일로 변환
            if self._current_instance_labels is not None and len(getattr(self, '_deleted_objects', set())) > 0:
                for obj_id in list(self._deleted_objects):
                    deleted_mask = (self._current_instance_labels == obj_id)
                    if np.sum(deleted_mask) > 0:
                        y_coords, x_coords = np.where(deleted_mask)
                        if len(y_coords) > 0:
                            cx = int(np.mean(x_coords) * getattr(self, 'display_scale', 1.0))
                            cy = int(np.mean(y_coords) * getattr(self, 'display_scale', 1.0))
                            cv2.line(preview_img, (cx-15, cy-15), (cx+15, cy+15), (255, 0, 0), 3)
                            cv2.line(preview_img, (cx+15, cy-15), (cx-15, cy+15), (255, 0, 0), 3)
                            cv2.line(preview_img, (cx-15, cy-15), (cx+15, cy+15), (255, 255, 255), 5)
                            cv2.line(preview_img, (cx+15, cy-15), (cx-15, cy+15), (255, 255, 255), 5)
                            cv2.line(preview_img, (cx-15, cy-15), (cx+15, cy+15), (255, 0, 0), 3)
                            cv2.line(preview_img, (cx+15, cy-15), (cx-15, cy+15), (255, 0, 0), 3)
            # Scale X 표시
            if self._current_scale_labels is not None and len(getattr(self, '_deleted_scale_objects', set())) > 0:
                for obj_id in list(self._deleted_scale_objects):
                    deleted_mask = (self._current_scale_labels == obj_id)
                    if np.sum(deleted_mask) > 0:
                        y_coords, x_coords = np.where(deleted_mask)
                        if len(y_coords) > 0:
                            cx = int(np.mean(x_coords) * getattr(self, 'display_scale', 1.0))
                            cy = int(np.mean(y_coords) * getattr(self, 'display_scale', 1.0))
                            cv2.line(preview_img, (cx-10, cy-10), (cx+10, cy+10), (255, 0, 0), 3)
                            cv2.line(preview_img, (cx+10, cy-10), (cx-10, cy+10), (255, 0, 0), 3)
                            cv2.line(preview_img, (cx-10, cy-10), (cx+10, cy+10), (255, 255, 255), 5)
                            cv2.line(preview_img, (cx+10, cy-10), (cx-10, cy+10), (255, 255, 255), 5)
                            cv2.line(preview_img, (cx-10, cy-10), (cx+10, cy+10), (255, 0, 0), 3)
                            cv2.line(preview_img, (cx+10, cy-10), (cx-10, cy+10), (255, 0, 0), 3)
        except Exception:
            pass
        
        print(f"최종 오버레이 이미지(디스플레이): {preview_img.shape}")
        return preview_img.astype(np.uint8)

    def show_preview_image(self, preview_image):
        """미리보기 이미지를 캔버스에 표시 (display_image 크기에 정렬, display_scale 불변)"""
        try:
            if preview_image is None:
                return
            
            # display_image 기준 크기 사용 (클릭 좌표 정합성 유지)
            if not hasattr(self, 'display_image') or self.display_image is None:
                self.update_display_image()
                if not hasattr(self, 'display_image') or self.display_image is None:
                    return
            display_h, display_w = self.display_image.shape[:2]
            
            # 미리보기를 display 크기에 맞춤
            resized_preview = cv2.resize(preview_image, (display_w, display_h))
            
            # 시드 마커 추가 (display_scale 유지)
            display_with_seeds = self.add_seed_markers(resized_preview.copy())
                
            # 분리 모드 강조: 전체에 옅은 어둡기 추가
            if getattr(self, 'split_mode_enabled', False):
                dim = (display_with_seeds * 0.92).astype(np.uint8)
            else:
                dim = display_with_seeds
            # PIL 변환 및 표시
            pil_image = Image.fromarray(dim)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # 캔버스 중앙에 표시 (크기 변경 없음)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
            
        except Exception as e:
            print(f"미리보기 표시 실패: {e}")

    def create_analysis_overlay(self, leaf_mask, scale_mask=None, stats: dict | None = None):
        """분석 결과 오버레이 생성 (Scale 마스크 포함)"""
        if not hasattr(self, 'working_image') or self.working_image is None:
            base_image = self.original_image.copy()
        else:
            base_image = self.working_image.copy()
        
        overlay = base_image.copy()
        try:
            overlay_font_scale = float(self.settings.get("overlay_font_scale", 0.45))
            overlay_font_thickness = int(self.settings.get("overlay_font_thickness", 1))
        except Exception:
            overlay_font_scale, overlay_font_thickness = 0.45, 1
        overlay_contour_thickness = self._get_contour_thickness()
        
        # 디버그: 마스크 상태 확인
        leaf_pixels = np.sum(leaf_mask) if leaf_mask is not None else 0
        scale_pixels = np.sum(scale_mask) if scale_mask is not None else 0
        print(f"오버레이 생성: leaf_mask={leaf_pixels}픽셀, scale_mask={scale_pixels}픽셀")
        
        # 잎 마스크 오버레이 (라벨 기반 색상) - 투명도 낮춤 (alpha=0.2 -> 80% 불투명)
        if leaf_mask is not None and np.sum(leaf_mask) > 0:
            if self._current_instance_labels is not None:
                include_ids = {int(i) for i in np.unique(self._current_instance_labels) if int(i) > 0}
                include_ids = {i for i in include_ids if i not in getattr(self, '_deleted_objects', set())}
                overlay = self._overlay_by_labels(overlay, self._current_instance_labels.astype(np.int32), include_ids, palette_type='leaf', alpha=0.2, contour_thickness=overlay_contour_thickness)
            else:
                overlay = self._overlay_instances(overlay, leaf_mask, palette_type='leaf', alpha=0.2, contour_thickness=overlay_contour_thickness)
        
        # Scale 마스크 오버레이 (라벨 기반 색상) - 투명도 낮춤 (alpha=0.2 -> 80% 불투명)
        if scale_mask is not None and np.sum(scale_mask) > 0:
            if self._current_scale_labels is not None:
                include_scale_ids = {int(i) for i in np.unique(self._current_scale_labels) if int(i) > 0}
                include_scale_ids = {i for i in include_scale_ids if i not in getattr(self, '_deleted_scale_objects', set())}
                overlay = self._overlay_by_labels(overlay, self._current_scale_labels.astype(np.int32), include_scale_ids, palette_type='scale', alpha=0.2, contour_thickness=overlay_contour_thickness)
            else:
                overlay = self._overlay_instances(overlay, scale_mask, palette_type='scale', alpha=0.2, contour_thickness=overlay_contour_thickness)
        
        # 객체 개수 표시 (stats 우선)
        if stats is None:
            stats = self._build_stats_from_masks(
                leaf_mask.astype(np.uint8) if leaf_mask is not None else None,
                scale_mask.astype(np.uint8) if scale_mask is not None else None,
                leaf_area_thresh=int(self.settings.get("min_object_area", 1)),
                scale_area_thresh=100
            )
        num_leaves = int(stats.get("active_leaf_count", 0))
        num_scales = int(stats.get("active_scale_count", 0))
        lines = [f"Leaves: {num_leaves}, Scales: {num_scales} (Preview)"]
        del_leaf = int(stats.get("deleted_leaf_count", 0))
        del_scale = int(stats.get("deleted_scale_count", 0))
        if del_leaf > 0 or del_scale > 0:
            lines.append(f"Deleted: Leaf {del_leaf}, Scale {del_scale}")
        overlay = self._draw_text_block(overlay, lines)
        
        print(f"오버레이 완료: {overlay.shape}, 마스크 적용됨")
        return overlay

    def show_preview_overlay(self, preview_image):
        """미리보기 오버레이를 캔버스에 표시"""
        try:
            if preview_image is None:
                print("오버레이 이미지가 None입니다")
                return
                
            print(f"캔버스 표시 시작: 이미지 크기 {preview_image.shape}")
            
            # 캔버스 크기에 맞춰 이미지 조정
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            print(f"캔버스 크기: {canvas_width}x{canvas_height}")
            
            if canvas_width <= 1 or canvas_height <= 1:
                print("캔버스 크기가 너무 작음")
                return
            
            h, w = preview_image.shape[:2]
            scale = min(canvas_width/w, canvas_height/h, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 이미지 크기 조정
            resized_preview = cv2.resize(preview_image, (new_w, new_h))
            
            # 시드 마커 추가 (기존 로직 재사용)
            if hasattr(self, 'display_scale'):
                temp_scale = self.display_scale
                self.display_scale = scale
                display_with_seeds = self.add_seed_markers(resized_preview.copy())
                self.display_scale = temp_scale
            else:
                display_with_seeds = resized_preview
            
            # PIL 변환 및 캔버스 표시
            print(f"PIL 변환 중: {display_with_seeds.shape}")
            pil_image = Image.fromarray(display_with_seeds)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            print("캔버스에 이미지 표시 중...")
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
            print("캔버스 표시 완료")
            
        except Exception as e:
            print(f"미리보기 표시 실패: {e}")

    def show_temporary_overlay(self, overlay_image):
        """임시 오버레이 이미지 표시"""
        # 현재 표시 이미지를 임시로 변경
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        h, w = overlay_image.shape[:2]
        scale = min(canvas_width/w, canvas_height/h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        temp_display = cv2.resize(overlay_image, (new_w, new_h))
        
        # PIL 이미지로 변환하여 표시
        pil_image = Image.fromarray(temp_display)
        temp_photo = ImageTk.PhotoImage(pil_image)
        
        # 캔버스에 임시 표시
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=temp_photo)
        
        # 임시 참조 유지 (가비지 컬렉션 방지)
        self.temp_photo = temp_photo
        
        # 3초 후 원래 이미지로 복원
        self.root.after(3000, self.update_display_image)

    def show_result_overlay(self, stats: dict | None = None):
        """결과 오버레이 표시 (삭제된 객체 제외). stats가 있으면 카운트에 재사용."""
        if not self.analysis_results:
            print("분석 결과가 없어 결과 표시 스킵")
            return
        
        print("최종 결과 오버레이 생성 시작...")
        
        # display_image를 기준으로 사용하여 크기 일관성 보장
        if not hasattr(self, 'display_image') or self.display_image is None:
            print("디스플레이 이미지가 없어 결과 표시 스킵")
            return
            
        # display_image를 오버레이 베이스로 사용
        overlay = self.display_image.copy()
        display_h, display_w = overlay.shape[:2]
        print(f"오버레이 베이스 크기: {display_w}x{display_h} (display_image 기준)")
        # 공통 두께 설정(초기 정의)
        overlay_contour_thickness = self._get_contour_thickness()
        
        # 삭제되지 않은 객체만 포함한 마스크 생성
        if self._current_instance_labels is not None:
            print(f"인스턴스 라벨맵 상태: {self._current_instance_labels.shape}, 최대 ID: {self._current_instance_labels.max()}")
            filtered_mask = self._create_filtered_mask()
        else:
            print("인스턴스 라벨맵이 None - 분석 결과의 leaf_mask를 직접 사용")
            # 분석 결과에서 직접 마스크 사용 (폴백)
            filtered_mask = self.analysis_results.get('leaf_mask', np.zeros((display_h, display_w), dtype=bool))
        
        if filtered_mask is not None and filtered_mask.size > 0:
            # 최적화: 캐싱된 리사이즈 사용
            leaf_mask_resized = self._cached_resize_mask(
                filtered_mask,
                (display_w, display_h),
                'result_leaf'
            )
            
            print(f"잎 마스크 리사이즈: {leaf_mask_resized.shape} -> {np.sum(leaf_mask_resized)}픽셀")
            print(f"오버레이 크기: {overlay.shape}")
            
            # 크기 일치 확인 (안전장치)
            if leaf_mask_resized.shape[:2] != overlay.shape[:2]:
                print(f"크기 불일치 감지: mask={leaf_mask_resized.shape[:2]}, overlay={overlay.shape[:2]}")
                return
            
            # 잎 마스크 오버레이: 라벨 기반 색상(병합/분리 후 ID 일관 반영) - 투명도 낮춤
            if np.sum(leaf_mask_resized) > 0:
                alpha_leaf = 0.3 if getattr(self, 'split_mode_enabled', False) else 0.2
                include_ids = set([int(i) for i in np.unique(self._current_instance_labels)] if self._current_instance_labels is not None else [1])
                include_ids = {i for i in include_ids if i > 0}
                # 삭제된 ID 제외
                include_ids = {i for i in include_ids if i not in getattr(self, '_deleted_objects', set())}
                if self._current_instance_labels is not None:
                    # 디스플레이 크기로 라벨맵 리사이즈
                    disp_labels = cv2.resize(self._current_instance_labels.astype(np.int32), (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                    highlight_leaf = set()
                    if getattr(self, 'merge_mode_enabled', False):
                        highlight_leaf |= {oid for (t, oid) in getattr(self, 'merge_selected', set()) if t == 'leaf'}
                    if getattr(self, 'delete_mode_enabled', False):
                        highlight_leaf |= {oid for (t, oid) in getattr(self, 'delete_selected', set()) if t == 'leaf'}
                    highlight_leaf = highlight_leaf if len(highlight_leaf) > 0 else None
                    overlay = self._overlay_by_labels(overlay, disp_labels, include_ids, palette_type='leaf', alpha=alpha_leaf, contour_thickness=overlay_contour_thickness,
                                                      highlight_ids=highlight_leaf)
                else:
                    overlay = self._overlay_instances(overlay, (leaf_mask_resized > 0), palette_type='leaf', alpha=alpha_leaf, contour_thickness=overlay_contour_thickness)
                print("잎 마스크(라벨 기반) 오버레이 적용")
            
            # 삭제된 Leaf 객체는 빨간색 X 표시 (인스턴스 라벨이 있을 때만)
            if self._current_instance_labels is not None:
                for obj_id in self._deleted_objects:
                    deleted_mask = (self._current_instance_labels == obj_id)
                    if np.sum(deleted_mask) > 0:
                        # 삭제된 객체 중심에 X 표시
                        y_coords, x_coords = np.where(deleted_mask)
                        if len(y_coords) > 0:
                            center_y = int(np.mean(y_coords) * self.display_scale)
                            center_x = int(np.mean(x_coords) * self.display_scale)
                            
                            # 빨간색 X 표시
                            cv2.line(overlay, (center_x-15, center_y-15), (center_x+15, center_y+15), (255, 0, 0), 3)
                            cv2.line(overlay, (center_x+15, center_y-15), (center_x-15, center_y+15), (255, 0, 0), 3)
                            
                            # X 주변에 흰색 테두리
                            cv2.line(overlay, (center_x-15, center_y-15), (center_x+15, center_y+15), (255, 255, 255), 5)
                            cv2.line(overlay, (center_x+15, center_y-15), (center_x-15, center_y+15), (255, 255, 255), 5)
                            cv2.line(overlay, (center_x-15, center_y-15), (center_x+15, center_y+15), (255, 0, 0), 3)
                            cv2.line(overlay, (center_x+15, center_y-15), (center_x-15, center_y+15), (255, 0, 0), 3)
            
            # 삭제된 Scale 객체도 빨간색 X 표시 
            if self._current_scale_labels is not None:
                for obj_id in self._deleted_scale_objects:
                    deleted_scale_mask = (self._current_scale_labels == obj_id)
                    if np.sum(deleted_scale_mask) > 0:
                        # 삭제된 Scale 객체 중심에 X 표시
                        y_coords, x_coords = np.where(deleted_scale_mask)
                        if len(y_coords) > 0:
                            center_y = int(np.mean(y_coords) * self.display_scale)
                            center_x = int(np.mean(x_coords) * self.display_scale)
                            
                            # 빨간색 X 표시 (Scale은 조금 더 작게)
                            cv2.line(overlay, (center_x-10, center_y-10), (center_x+10, center_y+10), (255, 0, 0), 3)
                            cv2.line(overlay, (center_x+10, center_y-10), (center_x-10, center_y+10), (255, 0, 0), 3)
                            
                            # X 주변에 흰색 테두리
                            cv2.line(overlay, (center_x-10, center_y-10), (center_x+10, center_y+10), (255, 255, 255), 5)
                            cv2.line(overlay, (center_x+10, center_y-10), (center_x-10, center_y+10), (255, 255, 255), 5)
                            cv2.line(overlay, (center_x-10, center_y-10), (center_x+10, center_y+10), (255, 0, 0), 3)
                            cv2.line(overlay, (center_x+10, center_y-10), (center_x-10, center_y+10), (255, 0, 0), 3)
        
        # Scale 마스크 오버레이 (반투명 빨간색)
        if 'scale_mask' in self.analysis_results and self.analysis_results['scale_mask'] is not None:
            # 삭제된 Scale 객체를 제외한 필터링된 마스크 사용
            if self._current_scale_labels is not None:
                filtered_scale_mask = self._create_filtered_scale_mask()
            else:
                # Scale 라벨맵이 없으면 원본 Scale 마스크 사용 (폴백)
                filtered_scale_mask = self.analysis_results['scale_mask']
            
            if filtered_scale_mask.size > 0:
                print(f"필터링된 Scale 마스크: {filtered_scale_mask.shape} -> {np.sum(filtered_scale_mask)}픽셀")
                
                # 최적화: 캐싱된 리사이즈 사용
                scale_mask_resized = self._cached_resize_mask(
                    filtered_scale_mask,
                    (display_w, display_h),
                    'result_scale'
                )
                
                print(f"Scale 마스크 리사이즈: {scale_mask_resized.shape} -> {np.sum(scale_mask_resized)}픽셀")
            else:
                scale_mask_resized = np.zeros((display_h, display_w), dtype=np.uint8)
                print("Scale 마스크가 비어있음 - 빈 마스크 사용")
            
            if np.sum(scale_mask_resized) > 0:
                # Leaf 마스크와 겹치는 영역 확인
                if 'leaf_mask_resized' in locals():
                    overlap = np.sum((leaf_mask_resized > 0) & (scale_mask_resized > 0))
                    print(f"   → Leaf와 Scale 겹치는 픽셀: {overlap}개")
                # Scale 마스크 오버레이 (라벨 기반) - 투명도 낮춤
                alpha_scale = 0.3 if getattr(self, 'split_mode_enabled', False) else 0.2
                if self._current_scale_labels is not None:
                    include_scale_ids = {int(i) for i in np.unique(self._current_scale_labels) if int(i) > 0}
                    include_scale_ids = {i for i in include_scale_ids if i not in getattr(self, '_deleted_scale_objects', set())}
                    disp_scale_labels = cv2.resize(self._current_scale_labels.astype(np.int32), (display_w, display_h), interpolation=cv2.INTER_NEAREST)
                    highlight_scale = set()
                    if getattr(self, 'merge_mode_enabled', False):
                        highlight_scale |= {oid for (t, oid) in getattr(self, 'merge_selected', set()) if t == 'scale'}
                    if getattr(self, 'delete_mode_enabled', False):
                        highlight_scale |= {oid for (t, oid) in getattr(self, 'delete_selected', set()) if t == 'scale'}
                    highlight_scale = highlight_scale if len(highlight_scale) > 0 else None
                    overlay = self._overlay_by_labels(overlay, disp_scale_labels, include_scale_ids, palette_type='scale', alpha=alpha_scale, contour_thickness=overlay_contour_thickness,
                                                      highlight_ids=highlight_scale)
                else:
                    overlay = self._overlay_instances(overlay, (scale_mask_resized > 0), palette_type='scale', alpha=alpha_scale, contour_thickness=overlay_contour_thickness)
                # Scale 윤곽선
                scale_contours_res, _ = cv2.findContours((scale_mask_resized > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, scale_contours_res, -1, (255, 0, 255), overlay_contour_thickness)
                print("Scale 마스크(다색) 오버레이 적용")
        else:
            print("Scale 마스크 없음 - 오버레이 스킵")
        
        # 잎 윤곽선(노란색) 추가 - 라벨맵 없을 때만
        try:
            if self._current_instance_labels is None and 'leaf_mask_resized' in locals() and leaf_mask_resized is not None:
                leaf_contours_res, _ = cv2.findContours((leaf_mask_resized > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, leaf_contours_res, -1, (255, 255, 0), overlay_contour_thickness)
        except Exception:
            pass
        
        # 상단 카운트 표기 (최종 결과에도 표시) - stats 우선 사용
        try:
            if stats is None and hasattr(self, '_compute_result_stats'):
                try:
                    stats = self._compute_result_stats()
                except Exception:
                    stats = None
            if stats is not None:
                num_leaves = int(stats.get("active_leaf_count", 0))
                num_scales = int(stats.get("active_scale_count", 0))
            else:
                # Leaf: 분석 결과 객체에서 삭제되지 않은 개수
                num_leaves = 0
                if hasattr(self, 'analysis_results') and self.analysis_results and 'objects' in self.analysis_results:
                    num_leaves = len([
                        obj for obj in self.analysis_results['objects']
                        if obj.get('id', 0) not in getattr(self, '_deleted_objects', set())
                    ])
                # Scale: 라벨맵이 있으면 삭제 상태 반영하여 개수 계산, 없으면 마스크 윤곽선 개수
                if self._current_scale_labels is not None:
                    num_scales = len([
                        sid for sid in np.unique(self._current_scale_labels)
                        if sid > 0 and sid not in getattr(self, '_deleted_scale_objects', set())
                    ])
                elif 'scale_mask_resized' in locals() and scale_mask_resized is not None:
                    scale_contours_res2, _ = cv2.findContours((scale_mask_resized > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    num_scales = len(scale_contours_res2)
                else:
                    num_scales = 0
            lines = [f"Leaves: {num_leaves}, Scales: {num_scales}"]
            if stats is not None:
                del_leaf = int(stats.get("deleted_leaf_count", 0))
                del_scale = int(stats.get("deleted_scale_count", 0))
                if del_leaf > 0 or del_scale > 0:
                    lines.append(f"Deleted: Leaf {del_leaf}, Scale {del_scale}")
            overlay = self._draw_text_block(overlay, lines)
        except Exception:
            pass
        
        # 시드 마커 추가 (이미 display 크기이므로 리사이즈 불필요)
        display_with_seeds = self.add_seed_markers(overlay.copy())
        
        # PIL 이미지로 변환하여 표시
        # 분리 모드 시각 강조: 약간 어둡게
        if getattr(self, 'split_mode_enabled', False):
            display_with_seeds = (display_with_seeds * 0.92).astype(np.uint8)
        pil_overlay = Image.fromarray(display_with_seeds.astype(np.uint8))
        self.photo = ImageTk.PhotoImage(pil_overlay)
        
        # 캔버스 크기 가져오기
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        print(f"캔버스 크기: {canvas_width}x{canvas_height}")
        print("최종 결과로 캔버스 업데이트")
        
        # 캔버스 업데이트
        # 분리 모드 강조: 살짝 dim 처리
        if getattr(self, 'split_mode_enabled', False):
            try:
                # 현재 포토 이미지를 가져오기 위해 다시 배열로 변환하여 dim 처리는 앞 단계에서 수행됨
                pass
            except Exception:
                pass
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
        print("최종 결과 표시 완료")
    
    # GrabCut 전환으로 자동 튜닝 기능 제거됨
