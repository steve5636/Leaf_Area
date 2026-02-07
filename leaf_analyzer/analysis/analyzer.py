#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaf Analyzer for Advanced Leaf Analyzer
분석 로직
"""

import time
import numpy as np
import cv2
from tkinter import messagebox
from skimage import measure
from scipy import ndimage

from ..core.morphology import MorphologicalAnalyzer
from ..processing.sam3_segmenter import Sam3Segmenter

class LeafAnalyzer:
    """잎 분석 믹스인 클래스"""

    def _apply_convex_hull_to_scale_mask(self, scale_mask: np.ndarray) -> np.ndarray:
        """Scale 마스크에 Convex Hull 적용하여 내부 hole 제거
        
        Args:
            scale_mask: 원본 Scale 마스크 (bool 또는 uint8)
            
        Returns:
            np.ndarray: Convex Hull이 적용된 마스크 (uint8)
        """
        if scale_mask is None or np.sum(scale_mask) == 0:
            return scale_mask
        
        # 연결 성분별로 Convex Hull 적용
        scale_labels, num_scales = ndimage.label(scale_mask)
        
        if num_scales == 0:
            return scale_mask.astype(np.uint8)
        
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
        
        return convex_mask

    def _generate_scale_mask_from_color_ratio(
        self,
        r_channel: np.ndarray,
        g_channel: np.ndarray,
        b_channel: np.ndarray,
        scale_mode: str,
        minR: int,
        ratR: float,
    ) -> np.ndarray:
        """색상 비율 기반 Scale 마스크 생성 (기본 분석 로직 재사용)."""
        if scale_mode == "blue":
            # 파란색 Scale 전용 파라미터
            minB = self.easy_params.get("minB", 80)
            ratB = self.easy_params.get("ratB", 1.3)
            blue_max_r = self.easy_params.get("blue_max_r", 150)
            blue_max_g = self.easy_params.get("blue_max_g", 150)
            
            # 파란색 Scale: B가 지배적이고 R, G가 낮은 영역
            scale_mask_raw = (
                (b_channel > minB) &                      # B 최소값
                (b_channel > r_channel * ratB) &          # B > R * ratB
                (b_channel > g_channel * ratB) &          # B > G * ratB
                (r_channel < blue_max_r) &                # R 억제 (흰색 배경 제외)
                (g_channel < blue_max_g)                  # G 억제 (초록 잎 제외)
            ).astype(np.uint8)
            print(f"파란색 Scale 검출: minB={minB}, ratB={ratB}, max_r={blue_max_r}, max_g={blue_max_g}")
        else:
            # 빨간색 Scale (기본): R이 지배적인 영역
            scale_mask_raw = (
                (r_channel > minR) &
                (r_channel > g_channel * ratR) &
                (r_channel > b_channel * ratR)
            ).astype(np.uint8)
            print(f"빨간색 Scale 검출: minR={minR}, ratR={ratR}")
        return scale_mask_raw

    def _filter_scale_mask(self, scale_mask: np.ndarray):
        """Scale 마스크 연결 성분 필터링 (노이즈 제거)."""
        if scale_mask is None:
            return scale_mask, None, 0
        scale_labels_raw, num_scales_raw = ndimage.label(scale_mask)
        # Scale 최소 크기 필터링 (노이즈 제거)
        scale_min_size = max(200, int(np.sum(scale_mask) * 0.05))  # 전체의 5% 또는 200픽셀
        if num_scales_raw > 0:
            scale_sizes = ndimage.sum(scale_mask, scale_labels_raw, range(1, num_scales_raw + 1))
            valid_scale_ids = [i + 1 for i, size in enumerate(scale_sizes) if size >= scale_min_size]
            if len(valid_scale_ids) > 0:
                scale_mask_filtered = np.isin(scale_labels_raw, valid_scale_ids)
                scale_labels, num_scales = ndimage.label(scale_mask_filtered)
                scale_mask = scale_mask_filtered.astype(np.uint8)
                print(f"Scale 필터링: {num_scales_raw}개 → {num_scales}개 (최소 {scale_min_size}픽셀)")
            else:
                scale_labels = scale_labels_raw
                num_scales = num_scales_raw
        else:
            scale_labels = scale_labels_raw
            num_scales = num_scales_raw
        return scale_mask, scale_labels, num_scales

    def _build_basic_result_message(self, leaf_count: int, leaf_area_px: int, scale_area_px: int, minG: int, ratG: float, ratGb: float) -> str:
        """기본 분석 결과 메시지 생성"""
        message = f"""기본 분석 완료!
    
탐지된 잎: {leaf_count}개
잎 면적: {leaf_area_px} 픽셀
스케일 면적: {scale_area_px} 픽셀
    
사용된 파라미터:
- 최소 녹색값: {minG}
- G/R 비율: {ratG}
- G/B 비율: {ratGb}
    
만약 결과가 만족스럽지 않으면 '고급 분석'을 사용하세요."""
        return message

    def _compute_result_stats(self) -> dict:
        """분석 결과 통계 계산 (활성/전체/삭제 포함)"""
        results = self.analysis_results or {}
        # Leaf
        all_leaf_objects = results.get('objects', [])
        total_leaf_count = len(all_leaf_objects)
        active_leaf_objects = [
            obj for obj in all_leaf_objects
            if obj.get('id', 0) not in self._deleted_objects
        ]
        active_leaf_count = len(active_leaf_objects)
        active_leaf_area_px = sum(obj.get('area', 0) for obj in active_leaf_objects)
        deleted_leaf_count = len(self._deleted_objects)
        deleted_scale_count = len(self._deleted_scale_objects)

        # Scale
        total_scale_count = 0
        active_scale_count = 0
        if self._current_scale_labels is not None:
            unique_scale_ids = [sid for sid in np.unique(self._current_scale_labels) if sid > 0]
            total_scale_count = len(unique_scale_ids)
            active_scale_count = len([sid for sid in unique_scale_ids if sid not in self._deleted_scale_objects])
        elif 'scale_mask' in results and results['scale_mask'] is not None:
            scale_mask_tmp = results['scale_mask'].astype(np.uint8)
            if scale_mask_tmp.size > 0 and np.sum(scale_mask_tmp) > 0:
                _, scale_labels_tmp = cv2.connectedComponents(scale_mask_tmp, connectivity=8)
                total_scale_count = int(np.max(scale_labels_tmp))
                # 활성 스케일을 알기 어려운 경우 전체와 동일로 표기
                active_scale_count = total_scale_count

        return {
            "total_leaf_count": total_leaf_count,
            "active_leaf_count": active_leaf_count,
            "active_leaf_area_px": active_leaf_area_px,
            "deleted_leaf_count": deleted_leaf_count,
            "total_scale_count": total_scale_count,
            "active_scale_count": active_scale_count,
            "deleted_scale_count": deleted_scale_count,
            "active_leaf_objects": active_leaf_objects,
        }

    def _build_analysis_result_message(self, stats: dict) -> str:
        """고급 분석 결과 메시지 생성"""
        results = self.analysis_results or {}
        method = results.get('method', 'advanced')
        if method == "basic_color_ratio":
            method_text = "기본 분석 (색상 비율)"
        elif method == "sam3_mixed":
            method_text = "혼합 분석 (SAM3)"
        else:
            method_text = "고급 분석"

        # 스케일 면적 정보
        scale_info = ""
        if 'scale_mask' in results and results['scale_mask'] is not None:
            scale_area = np.sum(results['scale_mask'])
            if scale_area > 0:
                scale_info = f"\n스케일 면적: {scale_area:.0f} 픽셀"
                if results.get('pixels_per_cm2', 1) > 0:
                    scale_cm2 = scale_area / results['pixels_per_cm2']
                    scale_info += f" ({scale_cm2:.2f} cm²)"

        # 삭제 정보
        deletion_info = ""
        if stats["deleted_leaf_count"] > 0 or stats.get("deleted_scale_count", 0) > 0:
            deletion_info = (
                f"\n\n삭제된 객체: Leaf {stats['deleted_leaf_count']}개, "
                f"Scale {stats.get('deleted_scale_count', 0)}개 (Ctrl+클릭으로 복원 가능)"
            )

        message = (
            f"분석 완료! [{method_text}]\n\n"
            f"전체 객체: Leaf {stats['total_leaf_count']}개, Scale {stats['total_scale_count']}개 (합계: {stats['total_leaf_count'] + stats['total_scale_count']}개)\n"
            f"활성 객체: Leaf {stats['active_leaf_count']}개, Scale {stats['active_scale_count']}개 (합계: {stats['active_leaf_count'] + stats['active_scale_count']}개)\n"
            f"활성 면적: {stats['active_leaf_area_px']:.0f} 픽셀 "
            f"({stats['active_leaf_area_px'] / max(results.get('pixels_per_cm2', 1), 1):.2f} cm²)"
            f"{scale_info}{deletion_info}\n\n"
            f"개별 잎 정보 (활성만):"
        )

        if method == "sam3_mixed":
            prompt = results.get("sam3_prompt", "")
            score_th = results.get("sam3_score_threshold", None)
            min_area_used = results.get("sam3_min_area_used", None)
            relaxed = results.get("sam3_min_area_relaxed", False)
            if prompt:
                message = f"[프롬프트: {prompt}]\n" + message
            if score_th is not None:
                message = f"[점수 임계값: {score_th}]\n" + message
            if min_area_used is not None:
                note = " (완화됨)" if relaxed else ""
                message = f"[최소 면적: {min_area_used}px{note}]\n" + message

        # 활성 객체만 표시 (상위 5개)
        active_leaf_objects = stats.get("active_leaf_objects", [])
        for i, obj in enumerate(active_leaf_objects[:5]):
            message += f"\n잎 {i+1}: {obj['area']:.0f}픽셀, 길이 {obj['length']:.1f}, 너비 {obj['width']:.1f}"

        if len(active_leaf_objects) > 5:
            message += f"\n... 외 {len(active_leaf_objects)-5}개"

        if stats["deleted_leaf_count"] > 0 or stats.get("deleted_scale_count", 0) > 0:
            message += (
                f"\n\n[Ctrl+클릭으로 숨김 상태 관리 중 - Leaf {stats['deleted_leaf_count']}, "
                f"Scale {stats.get('deleted_scale_count', 0)}]"
            )
        return message
    def analyze_image(self, forced: bool = False):
        """[개선] 간소화된 GrabCut 기반 이미지 분석"""
        print("analyze_image() 시작")
        now = time.monotonic()
        if not forced and (now - getattr(self, '_last_analyze_ts', 0.0)) < getattr(self, '_analyze_cooldown_seconds', 0.75):
            return
        self._last_analyze_ts = now
        if getattr(self, 'is_analyzing', False):
            return
        self.is_analyzing = True
        try:
            if hasattr(self, 'analyze_button') and self.analyze_button:
                self.analyze_button.configure(state="disabled")
        except Exception:
            pass
        try:
            if self.original_image is None:
                messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
                self._safe_refocus()  # messagebox 후 포커스 관리
                return
            if len(self.seed_manager.seeds.get("leaf", [])) < self.settings["min_seeds_required"].get("leaf", 1):
                messagebox.showwarning("경고", "잎(leaf) 시드를 더 추가해주세요.")
                self._safe_refocus()  # messagebox 후 포커스 관리
                return
            
            # 1) 세그멘테이션으로 전체 잎 영역 추정
            print("[1/2] 전체 잎 영역 분할 중...")
            coarse_leaf_mask = self.generate_leaf_mask()
    
            # 2) 연결 성분 분석으로 개별 잎 분리
            final_instance_labels, num_objects = measure.label(coarse_leaf_mask, connectivity=2, return_num=True)
            print(f"   ↳ 검출된 객체: {num_objects}개")
            
            final_leaf_mask = coarse_leaf_mask
            current_label_id = num_objects
    
            # 3) 형태 분석
            print(f"[2/2] 최종 {current_label_id}개 잎 형태 분석 중...")
            leaf_objects = []
            for leaf_id in range(1, current_label_id + 1):
                single_leaf_mask = (final_instance_labels == leaf_id)
                if np.sum(single_leaf_mask) < self.settings["min_object_area"]:
                    continue
                # 홀(내부) 반영 분석
                obj_data = MorphologicalAnalyzer.analyze_mask_with_holes(single_leaf_mask)
                obj_data["id"] = leaf_id
                leaf_objects.append(obj_data)
    
            # Scale 마스크 생성 (Scale seed 유무에 따라 조건부 실행)
            scale_seeds = self.seed_manager.seeds.get("scale", [])
            scale_mask = None  # 없는 경우도 있으므로 기본값을 명시적으로 설정
            if len(scale_seeds) > 0:
                print(f"   → Scale seed {len(scale_seeds)}개 기반 Scale 마스크 생성")
                scale_mask = self._generate_scale_mask_from_seeds(scale_seeds)
            else:
                print("   → Scale seed 없음 - Scale 검출 스킵")
                
            scale_area_pixels = np.sum(scale_mask) if scale_mask is not None else 0
            
            total_leaf_area_pixels = sum(obj["area"] for obj in leaf_objects)
            
            # Scale 기반 실제 면적 계산
            if scale_area_pixels > 0:
                # 기본 4cm² 스케일 가정 (Easy Leaf Area 호환)
                scale_area_cm2 = 4.0  # cm²
                pixels_per_cm2 = scale_area_pixels / scale_area_cm2
                total_leaf_area_cm2 = total_leaf_area_pixels / pixels_per_cm2
                print(f"   → Scale 면적: {scale_area_pixels}px = {scale_area_cm2}cm² (비율: {pixels_per_cm2:.1f}px/cm²)")
            else:
                pixels_per_cm2 = 1
                total_leaf_area_cm2 = 0
                print("   → Scale 검출 안됨 (픽셀 단위로 표시)")
            # 객체 선택적 삭제를 위한 인스턴스 라벨맵 저장
            self._current_instance_labels = final_instance_labels
            
            # Scale 객체도 개별 삭제 가능하도록 라벨맵 생성
            if scale_mask is not None and np.sum(scale_mask) > 0:
                # Scale 마스크에 연결 성분 분석 적용
                scale_num_labels, scale_labels = cv2.connectedComponents(
                    scale_mask.astype(np.uint8), connectivity=8
                )
                self._current_scale_labels = scale_labels
                print(f"   → Scale 개별 객체 라벨맵 생성: {scale_num_labels - 1}개 객체")
            else:
                self._current_scale_labels = None
                print("   → Scale 객체 없음 - 라벨맵 생성 스킵")
            
            self.analysis_results = {
                "total_objects": len(leaf_objects),
                "total_leaf_area_pixels": total_leaf_area_pixels,
                "total_leaf_area_cm2": total_leaf_area_cm2,
                "pixels_per_cm2": pixels_per_cm2,
                "objects": leaf_objects,
                "leaf_mask": final_leaf_mask,
                "scale_mask": scale_mask,
                "method": "advanced"  # 분석 방법 표시
            }
            print(f"   ↳ final instances: {len(leaf_objects)}, total_pixels: {total_leaf_area_pixels}")
            
            # 미리보기는 실시간 파라미터 조정에서만 사용
            # 분석 완료 후에는 최종 결과만 표시
            self.show_analysis_results()
        finally:
            self.is_analyzing = False
            try:
                if hasattr(self, 'analyze_button') and self.analyze_button:
                    self.analyze_button.configure(state="normal")
            except Exception:
                pass

    def mixed_analyze_sam3(self):
        """SAM3 기반 혼합 분석"""
        print("mixed_analyze_sam3() 시작")
        if self.original_image is None:
            messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
            self._safe_refocus()
            return
        if getattr(self, 'is_analyzing', False):
            return
        self.is_analyzing = True
        try:
            if hasattr(self, 'sam3_analyze_button') and self.sam3_analyze_button:
                self.sam3_analyze_button.configure(state="disabled")
        except Exception:
            pass

        try:
            img = self.working_image if hasattr(self, 'working_image') and self.working_image is not None else self.original_image
            prompt_var = getattr(self, "sam3_prompt_var", None)
            prompt = prompt_var.get() if prompt_var else "leaf"
            score_var = getattr(self, "sam3_score_threshold_var", None)
            try:
                score_threshold = float(score_var.get()) if score_var else 0.4
            except Exception:
                score_threshold = 0.4
            max_instances = int(self.settings.get("sam3_max_instances", 100))

            segmenter = getattr(self, "_sam3_segmenter", None)
            if segmenter is None:
                segmenter = Sam3Segmenter()
                self._sam3_segmenter = segmenter

            segments = segmenter.segment_image(
                img,
                prompt=prompt,
                score_threshold=score_threshold,
                max_instances=max_instances,
            )
            segments_count = len(segments)
            if not segments:
                messagebox.showwarning(
                    "SAM3 결과 없음",
                    "유효한 마스크가 없습니다.\n"
                    "• 키워드(프롬프트)를 변경하거나\n"
                    "• 점수 임계값을 낮춰보세요."
                )
                self._safe_refocus()
                return

            h, w = img.shape[:2]
            instance_labels = np.zeros((h, w), dtype=np.int32)
            used = np.zeros((h, w), dtype=bool)
            leaf_objects = []
            min_area = int(self.settings.get("min_object_area", 1000))
            relaxed_min_area = max(100, int(h * w * 0.00005))
            if relaxed_min_area >= min_area:
                relaxed_min_area = max(50, int(min_area * 0.2))
            min_area_used = min_area
            relaxed = False

            current_id = 0
            for seg in segments:
                mask = seg.get("mask", None)
                if mask is None:
                    continue
                mask = np.asarray(mask).astype(bool)
                if mask.shape[:2] != (h, w):
                    continue
                mask = mask & (~used)
                if int(np.sum(mask)) < min_area:
                    continue
                current_id += 1
                instance_labels[mask] = current_id
                used |= mask
                obj_data = MorphologicalAnalyzer.analyze_mask_with_holes(mask)
                obj_data["id"] = current_id
                obj_data["score"] = float(seg.get("score", 0.0))
                leaf_objects.append(obj_data)

            if len(leaf_objects) == 0:
                # 면적 기준 완화 재시도
                relaxed = True
                min_area_used = relaxed_min_area
                current_id = 0
                used = np.zeros((h, w), dtype=bool)
                instance_labels = np.zeros((h, w), dtype=np.int32)
                leaf_objects = []
                for seg in segments:
                    mask = seg.get("mask", None)
                    if mask is None:
                        continue
                    mask = np.asarray(mask).astype(bool)
                    if mask.shape[:2] != (h, w):
                        continue
                    mask = mask & (~used)
                    if int(np.sum(mask)) < min_area_used:
                        continue
                    current_id += 1
                    instance_labels[mask] = current_id
                    used |= mask
                    obj_data = MorphologicalAnalyzer.analyze_mask_with_holes(mask)
                    obj_data["id"] = current_id
                    obj_data["score"] = float(seg.get("score", 0.0))
                    leaf_objects.append(obj_data)

                if len(leaf_objects) == 0:
                    # 최종 폴백: 상위 스코어 마스크만 유지
                    current_id = 0
                    used = np.zeros((h, w), dtype=bool)
                    instance_labels = np.zeros((h, w), dtype=np.int32)
                    leaf_objects = []
                    for seg in segments:
                        mask = seg.get("mask", None)
                        if mask is None:
                            continue
                        mask = np.asarray(mask).astype(bool)
                        if mask.shape[:2] != (h, w):
                            continue
                        mask = mask & (~used)
                        if int(np.sum(mask)) == 0:
                            continue
                        current_id += 1
                        instance_labels[mask] = current_id
                        used |= mask
                        obj_data = MorphologicalAnalyzer.analyze_mask_with_holes(mask)
                        obj_data["id"] = current_id
                        obj_data["score"] = float(seg.get("score", 0.0))
                        leaf_objects.append(obj_data)
                        if current_id >= min(3, max_instances):
                            break

                if len(leaf_objects) == 0:
                    messagebox.showwarning(
                        "SAM3 결과 없음",
                        f"면적 기준을 만족하는 객체가 없습니다.\n"
                        f"현재 최소 면적: {min_area}px\n"
                        f"검출 마스크 수: {segments_count}"
                    )
                    self._safe_refocus()
                    return

            # Scale 마스크 (SAM3 텍스트 프롬프트 기반)
            scale_color = getattr(self, 'scale_color_var', None)
            scale_mode = scale_color.get() if scale_color else "red"
            scale_prompt = "red square" if scale_mode == "red" else "blue square"
            scale_segments = segmenter.segment_image(
                img,
                prompt=scale_prompt,
                score_threshold=score_threshold,
                max_instances=max_instances,
            )
            scale_mask = None
            scale_labels = None
            if scale_segments:
                scale_mask_raw = np.zeros((h, w), dtype=np.uint8)
                for seg in scale_segments:
                    mask = seg.get("mask", None)
                    if mask is None:
                        continue
                    mask = np.asarray(mask).astype(bool)
                    if mask.shape[:2] != (h, w):
                        continue
                    scale_mask_raw |= mask.astype(np.uint8)
                if np.sum(scale_mask_raw) > 0:
                    scale_mask = self._apply_convex_hull_to_scale_mask(scale_mask_raw)
                    scale_mask, scale_labels, _ = self._filter_scale_mask(scale_mask)

            total_leaf_area_pixels = int(sum(obj.get("area", 0) for obj in leaf_objects))
            scale_area_pixels = int(np.sum(scale_mask)) if scale_mask is not None else 0

            if scale_area_pixels > 0:
                scale_area_cm2 = float(self.settings.get("scale_area_cm2", 4.0))
                if scale_area_cm2 <= 0:
                    scale_area_cm2 = 4.0
                pixels_per_cm2 = scale_area_pixels / scale_area_cm2
                total_leaf_area_cm2 = total_leaf_area_pixels / pixels_per_cm2
            else:
                pixels_per_cm2 = 1
                total_leaf_area_cm2 = 0

            self._current_instance_labels = instance_labels
            if scale_mask is not None and np.sum(scale_mask) > 0:
                self._current_scale_labels = scale_labels
            else:
                self._current_scale_labels = None

            self.analysis_results = {
                "total_objects": len(leaf_objects),
                "total_leaf_area_pixels": total_leaf_area_pixels,
                "total_leaf_area_cm2": total_leaf_area_cm2,
                "pixels_per_cm2": pixels_per_cm2,
                "objects": leaf_objects,
                "leaf_mask": (instance_labels > 0),
                "scale_mask": (scale_mask > 0) if scale_mask is not None else None,
                "method": "sam3_mixed",
                "sam3_prompt": prompt,
                "sam3_score_threshold": score_threshold,
                "sam3_scale_prompt": scale_prompt,
                "sam3_scale_segments": len(scale_segments) if scale_segments else 0,
                "sam3_min_area_used": min_area_used,
                "sam3_min_area_relaxed": relaxed,
            }

            # 즉시 오버레이 표시
            stats = self._compute_result_stats()
            self.show_result_overlay(stats)
            # 결과 메시지 표시
            message = self._build_analysis_result_message(stats)
            if relaxed:
                message += f"\n\n[안내] 최소 면적 기준을 {min_area_used}px로 완화했습니다."
            messagebox.showinfo("분석 결과", message)
            self._safe_refocus()
            print(f"혼합 분석 완료: {len(leaf_objects)}개 잎 검출")
        except Exception as e:
            messagebox.showerror(
                "SAM3 오류",
                "SAM3 추론 실패:\n"
                f"{e}\n\n"
                "확인 사항:\n"
                "• PR #173 브랜치 적용 여부\n"
                "• MPS fallback 활성화(PYTORCH_ENABLE_MPS_FALLBACK=1)\n"
                "• einops/pycocotools 설치"
            )
            self._safe_refocus()
            return
        finally:
            self.is_analyzing = False
            try:
                if hasattr(self, 'sam3_analyze_button') and self.sam3_analyze_button:
                    self.sam3_analyze_button.configure(state="normal")
            except Exception:
                pass
        
    # 구형 색상 모델 마스크 생성 함수 제거됨

    def preview_analysis(self):
        """실시간 미리보기 (GrabCut 결과)"""
        if not hasattr(self, 'preview_enabled') or not self.preview_enabled.get():
            return
            
        if self.original_image is None:
            return
            
        print("실시간 미리보기 업데이트 중...")
        
        try:
            # 선택된 세그멘테이션으로 마스크 생성
            leaf_mask = self.generate_leaf_mask()
            
            if leaf_mask is None or np.sum(leaf_mask) == 0:
                print("시드가 부족하거나 마스크가 비어있어 미리보기를 건너뜁니다.")
                return
            
            print(f"생성된 마스크: {np.sum(leaf_mask)}픽셀")
            
            processed_mask = leaf_mask
            print(f"후처리 후 마스크: {np.sum(processed_mask)}픽셀")
            
            # Scale 마스크 생성 (시드가 있을 때만)
            scale_seeds = self.seed_manager.seeds.get("scale", [])
            if len(scale_seeds) > 0:
                print(f"   → 실시간 미리보기: Scale seed {len(scale_seeds)}개 검출")
                scale_mask = self._generate_scale_mask_from_seeds(scale_seeds)
            else:
                print("   → 실시간 미리보기: Scale seed 없음 - Scale 검출 스킵")
                scale_mask = None
            
            # 미리보기 오버레이 생성 및 디버그
            print("미리보기 오버레이 생성 시작...")
            preview_overlay = self.create_analysis_overlay(processed_mask, scale_mask)
            
            if preview_overlay is not None:
                print(f"오버레이 이미지 크기: {preview_overlay.shape}")
            
            # 캔버스에 표시
            self.show_preview_overlay(preview_overlay)
            
            print("실시간 미리보기 업데이트 완료!")
            
        except Exception as e:
            print(f"미리보기 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()

    def basic_analyze(self):
        """기본 분석: Easy Leaf Area 방식의 빠른 색상 기반 분할 (elaMac2024.py 로직)"""
        print("기본 분석 (elaMac2024.py 로직) 시작...")
        
        if self.original_image is None:
            messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
            self._safe_refocus()  # messagebox 후 포커스 관리
            return
            
        try:
            # RGB 이미지 준비 (resize 제외)
            img = self.working_image if hasattr(self, 'working_image') and self.working_image is not None else self.original_image
            h, w = img.shape[:2]
            print(f"원본 이미지 크기: {w}x{h}")
            
            # ========== 자동 파라미터 추정 ==========
            # 수동 조정하지 않은 경우 자동 추정 수행
            if not getattr(self, '_user_manually_adjusted_params', False):
                print("자동 파라미터 추정 수행 중...")
                
                # 시드가 충분하면 시드 기반, 아니면 이미지 전체 분석
                leaf_seeds = self.seed_manager.seeds.get("leaf", []) if hasattr(self, 'seed_manager') else []
                
                if len(leaf_seeds) >= 3:
                    estimated = self._estimate_params_from_seeds()
                else:
                    estimated = self._estimate_params_from_image()
                
                # 추정된 파라미터 적용
                if estimated:
                    self.easy_params.update(estimated)
                    
                    # UI 레이블 업데이트
                    if hasattr(self, 'easy_params_label'):
                        self.easy_params_label.configure(
                            text=f"G>{self.easy_params['minG']}, G/R>{self.easy_params['ratG']:.2f}, G/B>{self.easy_params['ratGb']:.2f}"
                        )
            
            # 파라미터 가져오기
            minG = self.easy_params["minG"]
            ratG = self.easy_params["ratG"]
            ratGb = self.easy_params["ratGb"]
            minR = self.easy_params["minR"]
            ratR = self.easy_params["ratR"]
            
            print(f"파라미터: minG={minG}, ratG={ratG}, ratGb={ratGb}, minR={minR}, ratR={ratR}")
            
            # RGB 채널 분리 (벡터화 연산)
            r_channel = img[:, :, 0].astype(np.float32)
            g_channel = img[:, :, 1].astype(np.float32)
            b_channel = img[:, :, 2].astype(np.float32)
            
            # 배경색 및 Scale 색상 설정 확인
            background_color = getattr(self, 'background_color_var', None)
            bg_mode = background_color.get() if background_color else "dark"
            scale_color = getattr(self, 'scale_color_var', None)
            scale_mode = scale_color.get() if scale_color else "red"
            print(f"배경색 모드: {bg_mode}, Scale 색상: {scale_mode}")
            
            # Leaf 마스크 생성 (배경색에 따라 다른 조건 적용)
            if bg_mode == "white":
                # 흰색 배경: 완화된 비율 + G-R/G-B 차이 조건
                min_diff = self.easy_params.get("min_green_diff", 10)
                white_mult = self.easy_params.get("white_ratio_mult", 0.9)
                ratG_adj = ratG * white_mult
                ratGb_adj = ratGb * white_mult
                leaf_mask_raw = (
                    (r_channel * ratG_adj < g_channel) & 
                    (b_channel * ratGb_adj < g_channel) & 
                    (g_channel > minG) &
                    (g_channel - r_channel > min_diff) &  # G-R 차이 조건
                    (g_channel - b_channel > min_diff)    # G-B 차이 조건
                ).astype(np.uint8)
                print(f"흰색 배경 모드: ratG={ratG_adj:.2f}, ratGb={ratGb_adj:.2f}, min_diff={min_diff}")
            else:
                # 검은색 배경: 표준 비율 조건
                dark_mult = self.easy_params.get("dark_ratio_mult", 1.25)
                ratG_adj = ratG * dark_mult
                ratGb_adj = ratGb * dark_mult
                leaf_mask_raw = (
                    (r_channel * ratG_adj < g_channel) & 
                    (b_channel * ratGb_adj < g_channel) & 
                    (g_channel > minG)
                ).astype(np.uint8)
                print(f"검은색 배경 모드: ratG={ratG_adj:.2f}, ratGb={ratGb_adj:.2f} (계수: {dark_mult})")
            
            # Scale 마스크 생성 (색상에 따라 다른 조건)
            if scale_mode == "blue":
                # 파란색 Scale 전용 파라미터
                minB = self.easy_params.get("minB", 80)
                ratB = self.easy_params.get("ratB", 1.3)
                blue_max_r = self.easy_params.get("blue_max_r", 150)
                blue_max_g = self.easy_params.get("blue_max_g", 150)
                
                # 파란색 Scale: B가 지배적이고 R, G가 낮은 영역
                scale_mask_raw = (
                    (b_channel > minB) &                      # B 최소값
                    (b_channel > r_channel * ratB) &          # B > R * ratB
                    (b_channel > g_channel * ratB) &          # B > G * ratB
                    (r_channel < blue_max_r) &                # R 억제 (흰색 배경 제외)
                    (g_channel < blue_max_g)                  # G 억제 (초록 잎 제외)
                ).astype(np.uint8)
                print(f"파란색 Scale 검출: minB={minB}, ratB={ratB}, max_r={blue_max_r}, max_g={blue_max_g}")
            else:
                # 빨간색 Scale (기본): R이 지배적인 영역
                scale_mask_raw = (
                    (r_channel > minR) & 
                    (r_channel > g_channel * ratR) & 
                    (r_channel > b_channel * ratR)
                ).astype(np.uint8)
                print(f"빨간색 Scale 검출: minR={minR}, ratR={ratR}")
            
            print(f"Raw Leaf 픽셀 수: {np.sum(leaf_mask_raw)}, Raw Scale 픽셀 수: {np.sum(scale_mask_raw)}")
            
            # ========== 형태학적 후처리 ==========
            # 잎 분할 문제 해결을 위해 Close 연산 적용
            # 잎맥, 밝은 부분 등으로 인한 분리를 연결
            
            # 이미지 크기에 비례한 커널 (더 크게)
            kernel_size = max(5, min(11, int(min(h, w) / 200)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Leaf: Close 연산으로 조각 연결 (iterations=2로 강화)
            leaf_mask = cv2.morphologyEx(leaf_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Scale: Convex Hull 적용하여 내부 hole 제거
            scale_mask = self._apply_convex_hull_to_scale_mask(scale_mask_raw)
            
            print(f"후처리 후 Leaf 픽셀 수: {np.sum(leaf_mask)}, Scale 픽셀 수: {np.sum(scale_mask)}")
            
            # 최소 크기 파라미터 (기본값 500으로 변경 - 연결 성분 분석 활성화)
            min_component = int(self.easy_params.get("min_component", 500))
            min_component_ratio = float(self.easy_params.get("min_component_ratio", 0.00025))
            min_component_dynamic = max(min_component, int(h * w * min_component_ratio))
            if min_component_dynamic != min_component:
                print(
                    f"min_component 동적 보정: {min_component} -> {min_component_dynamic} "
                    f"(ratio={min_component_ratio})"
                )
            min_component = min_component_dynamic
            
            # elaMac2024.py 로직 (144-215번 줄):
            # minPsize <= 10이면 연결 성분 분석 하지 않음
            if min_component > 10:
                print(f"연결 성분 분석 활성화 (min_component={min_component})")
                
                # ndimage.label을 사용한 연결 성분 분석 (elaMac2024.py 104번 줄)
                labels, num_leaves = ndimage.label(leaf_mask)
                print(f"초기 검출된 잎 객체 수: {num_leaves}")
                
                # Blob 분석 (elaMac2024.py 182-210번 줄)
                blobhist = ndimage.measurements.histogram(labels, 1, num_leaves, num_leaves)
                
                # 최소 크기 이상의 객체만 유지
                leaf_objects = []
                total_leaf_area = 0
                valid_label_id = 0
                
                for blob_id in range(1, num_leaves + 1):
                    blob_size = blobhist[blob_id - 1]
                    
                    if blob_size > min_component:
                        valid_label_id += 1
                        component_mask = (labels == blob_id)
                        
                        # 형태 분석 (홀 포함)
                        obj_data = MorphologicalAnalyzer.analyze_mask_with_holes(component_mask)
                        obj_data["id"] = valid_label_id
                        leaf_objects.append(obj_data)
                        total_leaf_area += int(obj_data.get("area", blob_size))
                
                print(f"최소 크기 필터링 후 잎 객체 수: {len(leaf_objects)}")
                
                # 최종 라벨맵 생성 (유효한 객체만 포함)
                final_labels = np.zeros_like(labels, dtype=np.int32)
                for i, obj in enumerate(leaf_objects):
                    # 원본 라벨 ID를 찾아서 새 ID로 매핑
                    for blob_id in range(1, num_leaves + 1):
                        if blobhist[blob_id - 1] > min_component and np.sum((labels == blob_id)) == obj["area"]:
                            final_labels[labels == blob_id] = obj["id"]
                            break
            else:
                # elaMac2024.py 211-215번 줄: 연결 성분 분석 하지 않음
                print("NO CONNECTED COMPONENT ANALYSIS")
                total_leaf_area = int(np.sum(leaf_mask))
                
                # 전체 Leaf 픽셀을 하나의 객체로 취급
                obj_data = MorphologicalAnalyzer.analyze_mask_with_holes(leaf_mask.astype(bool))
                obj_data["id"] = 1
                leaf_objects = [obj_data]
                
                # 라벨맵: 모든 leaf 픽셀에 ID 1 할당
                final_labels = leaf_mask.astype(np.int32)
                
                print(f"전체 Leaf를 단일 객체로 처리: {total_leaf_area}픽셀")
            
            # Scale 연결 성분 분석 + 크기 필터링
            scale_labels_raw, num_scales_raw = ndimage.label(scale_mask)
            
            # Scale 최소 크기 필터링 (노이즈 제거)
            # 스케일은 보통 큰 단일 객체이므로 작은 조각들은 노이즈
            scale_min_size = max(200, int(np.sum(scale_mask) * 0.05))  # 전체의 5% 또는 200픽셀
            
            # 가장 큰 스케일 객체만 유지 (또는 상위 몇 개)
            if num_scales_raw > 0:
                scale_sizes = ndimage.sum(scale_mask, scale_labels_raw, range(1, num_scales_raw + 1))
                valid_scale_ids = [i + 1 for i, size in enumerate(scale_sizes) if size >= scale_min_size]
                
                if len(valid_scale_ids) > 0:
                    # 유효한 스케일만 남기기
                    scale_mask_filtered = np.isin(scale_labels_raw, valid_scale_ids)
                    scale_labels, num_scales = ndimage.label(scale_mask_filtered)
                    scale_mask = scale_mask_filtered.astype(np.uint8)
                    print(f"Scale 필터링: {num_scales_raw}개 → {num_scales}개 (최소 {scale_min_size}픽셀)")
                else:
                    scale_labels = scale_labels_raw
                    num_scales = num_scales_raw
            else:
                scale_labels = scale_labels_raw
                num_scales = num_scales_raw
            
            scale_area = np.sum(scale_mask)
            
            # 결과 저장
            self.analysis_results = {
                "total_objects": len(leaf_objects),
                "total_leaf_area_pixels": total_leaf_area,
                "total_scale_area_pixels": scale_area,
                "total_leaf_area_cm2": 0,  # 스케일 기반 계산 필요
                "pixels_per_cm2": 1,
                "objects": leaf_objects,
                "leaf_mask": (leaf_mask > 0),
                "scale_mask": (scale_mask > 0),
                "method": "basic_color_ratio",
                "instance_labels": final_labels  # 객체 선택용 라벨맵
            }
            
            # 기본 분석용 인스턴스 라벨맵 저장
            self._current_instance_labels = final_labels
            
            # Scale 객체도 개별 삭제 가능하도록 라벨맵 생성
            if np.sum(scale_mask) > 0:
                self._current_scale_labels = scale_labels
                print(f"   → Scale 개별 객체 라벨맵 생성: {num_scales}개 객체")
            else:
                self._current_scale_labels = None
                print("   → Scale 객체 없음 - 라벨맵 생성 스킵")
            
            # 결과 표시
            message = self._build_basic_result_message(
                leaf_count=len(leaf_objects),
                leaf_area_px=total_leaf_area,
                scale_area_px=scale_area,
                minG=minG,
                ratG=ratG,
                ratGb=ratGb
            )
            
            messagebox.showinfo("기본 분석 결과", message)
            self._safe_refocus()  # messagebox 후 포커스 관리
            
            # 결과 시각화
            self.show_result_overlay()
            
            print(f"기본 분석 완료: {len(leaf_objects)}개 잎 검출")
            
        except Exception as e:
            print(f"기본 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("오류", f"기본 분석 중 오류가 발생했습니다:\n{e}")
            self._safe_refocus()  # messagebox 후 포커스 관리

    def show_analysis_results(self, show_message: bool = True):
        """분석 결과 표시"""
        if not self.analysis_results:
            return

        stats = self._compute_result_stats()
        if show_message:
            message = self._build_analysis_result_message(stats)
            messagebox.showinfo("분석 결과", message)
            self._safe_refocus()  # messagebox 후 포커스 관리

        # 결과 이미지 표시
        self.show_result_overlay(stats)

    def show_result_overlay(self, stats: dict | None = None):
        return super().show_result_overlay(stats)
