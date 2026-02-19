#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaf Analyzer for Advanced Leaf Analyzer
분석 로직
"""

import numpy as np
import cv2
from tkinter import messagebox
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
        scale_area_cm2_setting = float(self.settings.get("scale_area_cm2", 4.0))
        has_scale = scale_area_px > 0 and scale_area_cm2_setting > 0
        if has_scale:
            pixels_per_cm2 = float(scale_area_px) / float(scale_area_cm2_setting)
            leaf_area_text = f"{(float(leaf_area_px) / max(pixels_per_cm2, 1e-9)):.2f} cm² ({leaf_area_px} 픽셀)"
            scale_area_text = f"{scale_area_cm2_setting:.2f} cm² ({scale_area_px} 픽셀)"
        else:
            leaf_area_text = f"{leaf_area_px} 픽셀 (Scale 없음: cm² 환산 불가)"
            scale_area_text = "미검출 (cm² 환산 불가)"

        message = f"""기본 분석 완료!
    
탐지된 잎: {leaf_count}개
활성 면적: {leaf_area_text}
Scale 면적: {scale_area_text}
    
사용된 파라미터:
- 최소 녹색값: {minG}
- G/R 비율: {ratG}
- G/B 비율: {ratGb}
    
필요하면 '혼합 분석 (SAM3)' 또는 'SAM3 누락 객체 추가'로 보정하세요."""
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
        """분석 결과 메시지 생성"""
        results = self.analysis_results or {}
        method = results.get('method', 'sam3_mixed')
        if method == "basic_color_ratio":
            method_text = "기본 분석 (색상 비율)"
        elif method == "sam3_mixed":
            method_text = "혼합 분석 (SAM3)"
        else:
            method_text = "분석"

        scale_area_px = 0.0
        if 'scale_mask' in results and results['scale_mask'] is not None:
            scale_area_px = float(np.sum(results['scale_mask']))
        pixels_per_cm2 = float(results.get("pixels_per_cm2", 0) or 0)
        has_scale = scale_area_px > 0 and pixels_per_cm2 > 0

        if has_scale:
            active_area_line = (
                f"활성 면적: {(stats['active_leaf_area_px'] / pixels_per_cm2):.2f} cm² "
                f"({stats['active_leaf_area_px']:.0f} 픽셀)"
            )
            scale_line = f"Scale 면적: {(scale_area_px / pixels_per_cm2):.2f} cm² ({scale_area_px:.0f} 픽셀)"
        else:
            active_area_line = f"활성 면적: {stats['active_leaf_area_px']:.0f} 픽셀 (Scale 없음: cm² 환산 불가)"
            scale_line = "Scale 면적: 미검출 (cm² 환산 불가)"

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
            f"{active_area_line}\n"
            f"{scale_line}"
            f"{deletion_info}"
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

        if stats["deleted_leaf_count"] > 0 or stats.get("deleted_scale_count", 0) > 0:
            message += (
                f"\n\n[Ctrl+클릭으로 숨김 상태 관리 중 - Leaf {stats['deleted_leaf_count']}, "
                f"Scale {stats.get('deleted_scale_count', 0)}]"
            )
        return message

    def _ensure_seed_buckets(self) -> dict:
        """시드 버킷 키(leaf/scale/background) 존재 보장."""
        if not hasattr(self, "seed_manager") or self.seed_manager is None:
            return {}
        seeds = getattr(self.seed_manager, "seeds", None)
        if not isinstance(seeds, dict):
            seeds = {}
            self.seed_manager.seeds = seeds
        for key in ("leaf", "scale", "background"):
            if key not in seeds or not isinstance(seeds.get(key), list):
                seeds[key] = list(seeds.get(key, [])) if isinstance(seeds.get(key, []), (list, tuple)) else []
        return seeds

    def _pick_correction_candidate(
        self,
        segments,
        existing_mask: np.ndarray,
        min_new_pixels: int,
        positive_points=None,
        negative_points=None,
    ):
        """기존 마스크와 겹치지 않는 신규 픽셀이 충분한 SAM3 후보를 선택.

        우선순위:
        1) positive point를 포함하는 후보
        2) negative point를 덜 포함하는 후보
        3) 신규 픽셀 면적/점수
        """
        if existing_mask is None:
            existing_mask = np.zeros((0, 0), dtype=bool)
        pos_points = list(positive_points or [])
        neg_points = list(negative_points or [])
        best = None
        for seg in segments:
            mask = np.asarray(seg.get("mask")).astype(bool)
            if mask.size == 0:
                continue
            h, w = mask.shape[:2]
            if existing_mask.shape != mask.shape:
                existing = np.zeros_like(mask, dtype=bool)
            else:
                existing = existing_mask
            area = int(np.sum(mask))
            if area <= 0:
                continue
            new_mask = mask & (~existing)
            new_area = int(np.sum(new_mask))
            if new_area < int(max(1, min_new_pixels)):
                continue
            overlap_ratio = float((area - new_area) / max(1, area))
            # 대부분이 기존 객체와 중복되면 신규 객체로 채택하지 않음
            if overlap_ratio >= 0.95:
                continue

            # positive point가 하나도 포함되지 않으면 제외 (위치 일관성 강화)
            pos_hits = 0
            for px, py in pos_points:
                x = int(np.clip(round(float(px)), 0, max(0, w - 1)))
                y = int(np.clip(round(float(py)), 0, max(0, h - 1)))
                if bool(mask[y, x]):
                    pos_hits += 1
            if len(pos_points) > 0 and pos_hits <= 0:
                continue

            neg_hits = 0
            for nx, ny in neg_points:
                x = int(np.clip(round(float(nx)), 0, max(0, w - 1)))
                y = int(np.clip(round(float(ny)), 0, max(0, h - 1)))
                if bool(mask[y, x]):
                    neg_hits += 1

            score = float(seg.get("score", 0.0))
            rank = (pos_hits, -neg_hits, new_area, score)
            if (best is None) or (rank > best["rank"]):
                best = {
                    "new_mask": new_mask,
                    "score": score,
                    "new_area": new_area,
                    "pos_hits": pos_hits,
                    "neg_hits": neg_hits,
                    "rank": rank,
                }
        return best

    def _recompute_analysis_summary(self):
        """analysis_results 요약 필드(면적/스케일 환산) 재계산."""
        if not isinstance(self.analysis_results, dict):
            return
        objects = self.analysis_results.get("objects", [])
        if not isinstance(objects, list):
            objects = []
            self.analysis_results["objects"] = objects
        total_leaf_area_pixels = int(sum(float(obj.get("area", 0.0)) for obj in objects))
        self.analysis_results["total_leaf_area_pixels"] = total_leaf_area_pixels
        self.analysis_results["total_objects"] = int(len(objects))

        scale_mask = self.analysis_results.get("scale_mask", None)
        scale_area_pixels = int(np.sum(scale_mask)) if scale_mask is not None else 0
        if scale_area_pixels > 0:
            scale_area_cm2 = float(self.settings.get("scale_area_cm2", 4.0))
            if scale_area_cm2 <= 0:
                scale_area_cm2 = 4.0
            pixels_per_cm2 = float(scale_area_pixels) / float(scale_area_cm2)
            total_leaf_area_cm2 = float(total_leaf_area_pixels) / max(pixels_per_cm2, 1e-9)
        else:
            pixels_per_cm2 = 1.0
            total_leaf_area_cm2 = 0.0
        self.analysis_results["pixels_per_cm2"] = pixels_per_cm2
        self.analysis_results["total_leaf_area_cm2"] = total_leaf_area_cm2

    def apply_sam3_seed_correction(self, show_message: bool = True) -> bool:
        """시드(포인트) 기반 SAM3 후보정: 누락 객체 1개 추가."""
        self._last_analysis_error = None
        if self.original_image is None:
            self._last_analysis_error = "image_not_loaded"
            if show_message:
                messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
                self._safe_refocus()
            return False
        if not isinstance(self.analysis_results, dict):
            self._last_analysis_error = "no_analysis_results"
            if show_message:
                messagebox.showwarning("후보정", "먼저 기본 분석 또는 혼합 분석(SAM3)을 실행하세요.")
                self._safe_refocus()
            return False
        if getattr(self, "is_analyzing", False):
            self._last_analysis_error = "analyze_in_progress"
            return False

        seeds = self._ensure_seed_buckets()
        leaf_pos = list(seeds.get("leaf", []))
        scale_pos = list(seeds.get("scale", []))
        bg_neg = list(seeds.get("background", []))

        has_leaf = len(leaf_pos) > 0
        has_scale = len(scale_pos) > 0
        if has_leaf and has_scale:
            self._last_analysis_error = "mixed_target_seeds"
            if show_message:
                messagebox.showwarning(
                    "후보정",
                    "Leaf와 Scale 양쪽 시드가 동시에 있습니다.\n한 대상만 남기고 다시 시도하세요."
                )
                self._safe_refocus()
            return False
        if (not has_leaf) and (not has_scale):
            self._last_analysis_error = "missing_positive_seed"
            if show_message:
                messagebox.showwarning(
                    "후보정",
                    "누락 객체를 추가하려면 Leaf 또는 Scale 시드를 1개 이상 찍어주세요."
                )
                self._safe_refocus()
            return False

        target = "leaf" if has_leaf else "scale"
        positive_points = leaf_pos if target == "leaf" else scale_pos

        try:
            score_var = getattr(self, "sam3_score_threshold_var", None)
            score_threshold = float(score_var.get()) if score_var else 0.4
        except Exception:
            score_threshold = 0.4
        score_threshold = float(np.clip(score_threshold, 0.0, 1.0))
        max_instances = int(self.settings.get("sam3_max_instances", 100))
        img = self.working_image if hasattr(self, "working_image") and self.working_image is not None else self.original_image
        h, w = img.shape[:2]

        if target == "leaf":
            prompt_var = getattr(self, "sam3_prompt_var", None)
            prompt = prompt_var.get() if prompt_var else "leaf"
        else:
            scale_color = getattr(self, "scale_color_var", None)
            scale_mode = scale_color.get() if scale_color else "red"
            prompt = "red square" if scale_mode == "red" else "blue square"

        points = [tuple(map(float, pt)) for pt in positive_points]
        labels = [1] * len(points)
        points.extend([tuple(map(float, pt)) for pt in bg_neg])
        labels.extend([0] * len(bg_neg))

        segmenter = getattr(self, "_sam3_segmenter", None)
        if segmenter is None:
            segmenter = Sam3Segmenter()
            self._sam3_segmenter = segmenter

        self.is_analyzing = True
        try:
            segments = segmenter.segment_image_with_points(
                img,
                prompt=prompt,
                points=points,
                point_labels=labels,
                score_threshold=score_threshold,
                max_instances=max_instances,
            )
        except Exception as e:
            err_text = str(e)
            if "MPS backend out of memory" in err_text:
                self._last_analysis_error = "sam3_correction_mps_oom"
                if show_message:
                    messagebox.showerror(
                        "SAM3 후보정 오류",
                        "MPS 메모리 할당 오류가 발생했습니다.\n\n"
                        "조치 방법:\n"
                        "• 추론 리사이즈 배율을 높인 뒤 다시 시도\n"
                        "• 앱을 재시작 후 재시도\n"
                        "• 필요 시 CPU 환경에서 실행"
                    )
                    self._safe_refocus()
            else:
                self._last_analysis_error = f"sam3_correction_error: {e}"
                if show_message:
                    messagebox.showerror(
                        "SAM3 후보정 오류",
                        f"후보정 중 오류가 발생했습니다:\n{e}"
                    )
                    self._safe_refocus()
            return False
        finally:
            self.is_analyzing = False

        if not segments:
            self._last_analysis_error = "sam3_correction_no_segments"
            if show_message:
                messagebox.showwarning(
                    "후보정",
                    "시드 기반 SAM3 결과가 없습니다.\n시드를 추가하거나 점수 임계값을 낮춰보세요."
                )
                self._safe_refocus()
            return False

        if target == "leaf":
            if self._current_instance_labels is not None and self._current_instance_labels.shape[:2] == (h, w):
                existing_mask = self._current_instance_labels > 0
                labels_map = self._current_instance_labels.astype(np.int32).copy()
            else:
                base_leaf_mask = np.asarray(self.analysis_results.get("leaf_mask")).astype(bool) if self.analysis_results.get("leaf_mask") is not None else np.zeros((h, w), dtype=bool)
                _, labels_map = cv2.connectedComponents(base_leaf_mask.astype(np.uint8), connectivity=8)
                labels_map = labels_map.astype(np.int32)
                existing_mask = labels_map > 0
            min_new_pixels = max(25, int(self.settings.get("min_object_area", 1000) * 0.15))
        else:
            if self._current_scale_labels is not None and self._current_scale_labels.shape[:2] == (h, w):
                existing_mask = self._current_scale_labels > 0
                labels_map = self._current_scale_labels.astype(np.int32).copy()
            else:
                base_scale_mask = np.asarray(self.analysis_results.get("scale_mask")).astype(bool) if self.analysis_results.get("scale_mask") is not None else np.zeros((h, w), dtype=bool)
                _, labels_map = cv2.connectedComponents(base_scale_mask.astype(np.uint8), connectivity=8)
                labels_map = labels_map.astype(np.int32)
                existing_mask = labels_map > 0
            min_new_pixels = max(15, int(self.settings.get("min_object_area", 1000) * 0.05))

        picked = self._pick_correction_candidate(
            segments,
            existing_mask,
            min_new_pixels=min_new_pixels,
            positive_points=positive_points,
            negative_points=bg_neg,
        )
        if picked is None:
            self._last_analysis_error = "sam3_correction_no_new_pixels"
            if show_message:
                messagebox.showwarning(
                    "후보정",
                    "시드 위치를 포함하는 신규 객체를 찾지 못했습니다.\n"
                    "• 누락 객체 내부에 Leaf/Scale 시드를 찍고\n"
                    "• 필요하면 배경(-) 시드를 더 추가해 다시 시도하세요."
                )
                self._safe_refocus()
            return False

        new_mask = picked["new_mask"]
        new_id = int(labels_map.max()) + 1
        labels_map[new_mask] = new_id

        if target == "leaf":
            self._current_instance_labels = labels_map
            new_obj = MorphologicalAnalyzer.analyze_mask_with_holes(new_mask)
            new_obj["id"] = new_id
            new_obj["score"] = float(picked["score"])
            objects = self.analysis_results.get("objects", [])
            if not isinstance(objects, list):
                objects = []
            objects.append(new_obj)
            self.analysis_results["objects"] = objects
            self.analysis_results["leaf_mask"] = (labels_map > 0)
            if hasattr(self, "_deleted_objects"):
                self._deleted_objects.discard(new_id)
        else:
            self._current_scale_labels = labels_map
            self.analysis_results["scale_mask"] = (labels_map > 0)
            if hasattr(self, "_deleted_scale_objects"):
                self._deleted_scale_objects.discard(new_id)

        self._recompute_analysis_summary()
        self.analysis_results["sam3_point_correction"] = True
        self.analysis_results["sam3_point_correction_target"] = target
        self.analysis_results["sam3_point_correction_new_area"] = int(picked["new_area"])
        self._reset_overlay_resize_cache()

        # 후보정 완료 후 시드 정리
        seeds["leaf"] = []
        seeds["scale"] = []
        seeds["background"] = []
        try:
            self.update_display_image()
            stats = self._compute_result_stats()
            self.show_result_overlay(stats)
        except Exception:
            pass

        if show_message:
            title = "Leaf 추가 완료" if target == "leaf" else "Scale 추가 완료"
            messagebox.showinfo(
                title,
                f"SAM3 후보정으로 {target} 객체 1개를 추가했습니다.\n"
                f"신규 면적: {int(picked['new_area'])} px\n"
                f"점수: {float(picked['score']):.3f}\n"
                f"시드 포함: +{int(picked.get('pos_hits', 0))}, -{int(picked.get('neg_hits', 0))}"
            )
            self._safe_refocus()
        return True

    def _reset_overlay_resize_cache(self):
        """분석 결과 갱신 직후 오버레이 리사이즈 캐시 초기화."""
        try:
            if hasattr(self, "_resize_cache"):
                self._resize_cache.clear()
            if hasattr(self, "_cache_version"):
                self._cache_version += 1
        except Exception:
            pass
    def analyze_image(self, forced: bool = False, show_message: bool = True, show_overlay: bool = True) -> bool:
        """호환용 엔트리포인트: 고급 분석 제거 후 SAM3 분석으로 위임."""
        return self.mixed_analyze_sam3(show_message=show_message, show_overlay=show_overlay)

    def mixed_analyze_sam3(self, show_message: bool = True, show_overlay: bool = True) -> bool:
        """SAM3 기반 혼합 분석"""
        print("mixed_analyze_sam3() 시작")
        self._last_analysis_error = None
        if self.original_image is None:
            self._last_analysis_error = "image_not_loaded"
            if show_message:
                messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
                self._safe_refocus()
            return False
        if getattr(self, 'is_analyzing', False):
            self._last_analysis_error = "analyze_in_progress"
            return False
        success = False
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
                self._last_analysis_error = "sam3_no_segments"
                if show_message:
                    messagebox.showwarning(
                        "SAM3 결과 없음",
                        "유효한 마스크가 없습니다.\n"
                        "• 키워드(프롬프트)를 변경하거나\n"
                        "• 점수 임계값을 낮춰보세요."
                    )
                    self._safe_refocus()
                return False

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
                    self._last_analysis_error = "sam3_no_objects_after_filter"
                    if show_message:
                        messagebox.showwarning(
                            "SAM3 결과 없음",
                            f"면적 기준을 만족하는 객체가 없습니다.\n"
                            f"현재 최소 면적: {min_area}px\n"
                            f"검출 마스크 수: {segments_count}"
                        )
                        self._safe_refocus()
                    return False

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
            self._reset_overlay_resize_cache()

            # 즉시 오버레이 표시
            stats = self._compute_result_stats()
            if show_overlay:
                self.show_result_overlay(stats)
            # 결과 메시지 표시
            if show_message:
                message = self._build_analysis_result_message(stats)
                if relaxed:
                    message += f"\n\n[안내] 최소 면적 기준을 {min_area_used}px로 완화했습니다."
                messagebox.showinfo("분석 결과", message)
                self._safe_refocus()
            success = True
            print(f"혼합 분석 완료: {len(leaf_objects)}개 잎 검출")
            return True
        except Exception as e:
            self._last_analysis_error = f"sam3_error: {e}"
            if show_message:
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
            return False
        finally:
            self.is_analyzing = False
            try:
                if hasattr(self, 'sam3_analyze_button') and self.sam3_analyze_button:
                    self.sam3_analyze_button.configure(state="normal")
            except Exception:
                pass
        return success
        
    # 구형 색상 모델 마스크 생성 함수 제거됨

    def basic_analyze(self, show_message: bool = True, show_overlay: bool = True) -> bool:
        """기본 분석: Easy Leaf Area 방식의 빠른 색상 기반 분할 (elaMac2024.py 로직)"""
        print("기본 분석 (elaMac2024.py 로직) 시작...")
        self._last_analysis_error = None
        
        if self.original_image is None:
            self._last_analysis_error = "image_not_loaded"
            if show_message:
                messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
                self._safe_refocus()  # messagebox 후 포커스 관리
            return False
            
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
            self._reset_overlay_resize_cache()
            
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
            
            if show_message:
                messagebox.showinfo("기본 분석 결과", message)
                self._safe_refocus()  # messagebox 후 포커스 관리
            
            # 결과 시각화
            if show_overlay:
                self.show_result_overlay()
            
            print(f"기본 분석 완료: {len(leaf_objects)}개 잎 검출")
            return True
            
        except Exception as e:
            print(f"기본 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            self._last_analysis_error = f"basic_analysis_error: {e}"
            if show_message:
                messagebox.showerror("오류", f"기본 분석 중 오류가 발생했습니다:\n{e}")
                self._safe_refocus()  # messagebox 후 포커스 관리
            return False

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
