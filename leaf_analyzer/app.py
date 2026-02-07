#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Leaf Area Analyzer - Main Application Class
GrabCut 기반 전경(잎) 분리 + 형태학 분석
"""

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# GUI 라이브러리
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
    from tkinter import ttk
    CTK_AVAILABLE = True
except ImportError:
    from tkinter import ttk
    CTK_AVAILABLE = False

# 이미지 처리 및 수치 계산
import cv2
import numpy as np
from skimage import measure
from scipy import ndimage

import networkx as nx
import time

# Core 모듈
from .core.seed_manager import SeedManager
from .core.morphology import MorphologicalAnalyzer
from .core.segmentation import GrabCutSegmenter

# Processing 모듈
from .processing.image_processor import ImageProcessor
from .processing.mask_generator import MaskGenerator
from .processing.overlay import OverlayManager

# Export 모듈
from .export.exporters import DataExporter
from .export.export_utils import ExportUtils

# GUI 모듈
from .gui.setup import GUISetup
from .gui.events import EventHandlers

# Analysis 모듈
from .analysis.analyzer import LeafAnalyzer

# Parameters 모듈
from .parameters.parameter_estimator import ParameterEstimator

# Operations 모듈
from .operations.object_operations import ObjectOperations

# ΔE2000 (가능하면 정밀 계산 모듈 사용)
try:
    from .utils.color_analysis_utils import DeltaE2000 as _DE2000
    def delta_e2000(a, b):
        return _DE2000.delta_e_2000(a, b)
except Exception:
    # 폴백: 간단한 CIE76
    def delta_e2000(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.sqrt(np.sum((a - b) ** 2)))

# 설정 (CustomTkinter가 사용 가능한 경우에만)
if CTK_AVAILABLE:
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

class AdvancedLeafAnalyzer(GUISetup, EventHandlers, LeafAnalyzer, ImageProcessor, MaskGenerator, OverlayManager, DataExporter, ExportUtils, ParameterEstimator, ObjectOperations):
    """고급 잎 분석기 메인 클래스"""
    
    def __init__(self):
        # 형태학 기본값 (GrabCut 이후 후처리)
        self.manual_settings = {
            "min_area": 1000,
            "morph_kernel": 5,
            "manual_preview": True
        }
        self.manual_params_visible = False
        self.manual_params_frame = None
        
        # 최적화: 마스크 리사이징 캐시
        self._resize_cache = {}
        self._cache_max_size = 10
        self._cache_version = 0  # 캐시 무효화용 버전
        
        # 최적화: Morphology 커널 캐시
        self._morph_kernels = {}
        
        # Easy Leaf Area 호환 파라미터
        # 기본 분석 파라미터 (명확한 배경 + 뚜렷한 초록 잎에 최적화)
        self.easy_params = {
            # 잎 검출 파라미터
            "minG": 25,           # 잎의 최소 녹색 RGB 값
            "ratG": 1.06,         # 잎 G/R 비율 (G > R * ratG)
            "ratGb": 1.08,        # 잎 G/B 비율 (G > B * ratGb)
            # 빨간 스케일 검출 파라미터
            "minR": 180,          # 스케일 최소 빨간색 RGB 값
            "ratR": 1.5,          # 스케일 R/G, R/B 비율
            # 파란 스케일 검출 파라미터
            "minB": 80,           # 스케일 최소 파란색 RGB 값
            "ratB": 1.3,          # 스케일 B/R, B/G 비율
            "blue_max_r": 150,    # 파란 스케일의 최대 R 값 (빨강 억제)
            "blue_max_g": 150,    # 파란 스케일의 최대 G 값 (초록 억제)
            # 필터링 파라미터
            "min_component": 500,  # 최소 컴포넌트 크기 (픽셀)
            # 배경색별 조정 계수
            "min_green_diff": 10,  # G-R, G-B 최소 차이 (흰색 배경 모드)
            "dark_ratio_mult": 1.25,  # 검은 배경 비율 계수
            "white_ratio_mult": 0.9,  # 흰색 배경 비율 계수
        }
        
        # 파라미터 수동 조정 플래그 (False = 자동 추정 활성화)
        self._user_manually_adjusted_params = False
        
        # 설정
        self.settings = {
            "preview_enabled": True,
            "min_object_area": 1000,      # 3000 → 1000으로 감소 (작은 잎도 검출)
            "morphology_kernel_size": 5,  # 3 → 5로 증가
            "inference_resize_divisor": 1.0,
            # 최소 시드 개수 요구사항 (필수 클래스에 적용)
            "min_seeds_required": {"leaf": 3},
            # GrabCut 경계 고정 비율(0.0~0.01)
            "grabcut_border_ratio": 0.005,  # 경계 고정 강화
            # 추가 파라미터
            "remove_border_touches": False,  # 경계 터치 제거 비활성화
            "bilateral_d": 9,                # bilateral filter 직경
            "bilateral_sigma_color": 75,     # 색상 시그마
            "bilateral_sigma_space": 75,     # 공간 시그마
            # 전처리(색상 분리 강화)
            "preprocess_enabled": True,
            "preprocess_method": "bilateral",  # bilateral|meanshift|none
            "pre_bilateral_d": 5,
            "pre_bilateral_sigma_color": 50,
            "pre_bilateral_sigma_space": 50,
            "pre_meanshift_sp": 10,
            "pre_meanshift_sr": 20,
            # 얇은 가지 보존용 적응형 후처리
            "thin_branch_dist_thresh": 2.4,   # 거리변환 중앙값(픽셀) 기준
            "thin_branch_kernel_scale": 0.6,  # 얇을 때 커널 스케일
            # 오버레이 스타일(글꼴/윤곽 두께) 통일 설정
            "overlay_font_scale": 0.45,
            "overlay_font_thickness": 1,
            "overlay_contour_thickness": 1,
            # Export용 스케일 면적(cm^2)
            "scale_area_cm2": 4.0
        }
        
        self.seed_manager = SeedManager()
        # GrabCut 세그멘터
        self.grabcut = GrabCutSegmenter()
        
        # GrabCut 전환 후 고급 색상 유틸 미사용
        self.adaptive_tuner = None
        self.background_suppressor = None
            
        # 이미지 데이터
        self.original_image = None
        self.original_image_full = None
        self.display_image = None
        self.hsv_image = None
        self.lab_image = None
        
        # 분석 결과
        self.current_masks = {"leaf": None, "scale": None, "background": None}
        self.analysis_results = None
        
        # 캐시된 원본 마스크 (필터링 전)
        self._cached_raw_mask = None
        self._cached_scale_mask = None
        self._last_seed_signature = None
        
        # 객체 선택적 삭제 시스템
        self._deleted_objects = set()  # 삭제된 Leaf 객체 ID 집합
        self._deleted_scale_objects = set()  # 삭제된 Scale 객체 ID 집합
        self._current_instance_labels = None  # 현재 Leaf 인스턴스 라벨맵
        self._current_scale_labels = None  # 현재 Scale 인스턴스 라벨맵
        self._object_deletion_enabled = True  # 객체 삭제 기능 활성화
        # 분석 재진입 방지 플래그
        self.is_analyzing = False
        # 버튼 참조 (초기값)
        self.analyze_button = None
        self.auto_tune_button = None
        # 분석 호출 디바운스
        self._last_analyze_ts = 0.0
        self._analyze_cooldown_seconds = 0.75
        # 내부 로그 저장 (콘솔 출력 비활성 기본)
        self.internal_logs: List[str] = []
        self.enable_console_log: bool = False
        
        # GUI 설정 (모든 변수 정의 후 마지막에 호출)
        self.setup_gui()

        # --- 분리 모드 상태 ---
        self.split_mode_enabled = False
        self.split_mode_points = []  # 워터셰드용 시드 두 점 [(x1,y1), (x2,y2)]
        self.split_selected_object = None  # ("leaf"|"scale", id)
        self._split_snapshot = None  # Undo용 (labels, objects) 백업
        # --- 병합 모드 상태 ---
        self.merge_mode_enabled = False
        self.merge_selected = set()  # { ("leaf"|"scale", id), ... }
        self._merge_snapshot = None  # Undo용 (instance_labels, scale_labels, analysis_results)
        # --- 삭제 모드 상태 ---
        self.delete_mode_enabled = False
        self.delete_selected = set()  # { ("leaf"|"scale", id), ... }
        # 마지막 분석 종류 (재탐색용)
        self._last_analysis_kind = None
        # 현재 적용중인 리사이즈 배율
        self._current_resize_divisor = 1.0

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
    
    def _safe_refocus(self):
        """대화상자 후 안전한 포커스 관리 (강화된 버전)
        
        목적:
        - 모든 버튼에서 포커스 해제 → Enter 키 중복 실행 방지
        - 캔버스로 포커스 이동
        - 일시적으로 Return 키 차단
        
        호출 시점:
        - filedialog 호출 직후
        - messagebox 호출 직후
        - 새 윈도우 닫힌 직후
        """
        try:
            # 1. Return 키 일시 차단 (500ms)
            self._block_return_key = True
            
            # 2. 캔버스로 포커스 이동 (버튼 포커스 해제)
            if hasattr(self, 'canvas') and self.canvas and self.canvas.winfo_exists():
                self.canvas.focus_set()
            
            # 3. 이벤트 큐 처리
            if hasattr(self, 'root') and self.root and self.root.winfo_exists():
                self.root.update_idletasks()
            
            # 4. 500ms 후 Return 키 차단 해제
            def _unblock():
                self._block_return_key = False
            if hasattr(self, 'root') and self.root and self.root.winfo_exists():
                self.root.after(500, _unblock)
                
        except Exception as e:
            # 포커스 설정 실패해도 프로그램은 계속 실행 (silent fail)
            self._block_return_key = False
    
    
    def setup_gui(self):
        return super().setup_gui()
    def _warn_if_ctk_missing(self):
        """CustomTkinter 미설치 시 간소화 UI 안내"""
        if CTK_AVAILABLE:
            return
        message = (
            "CustomTkinter이 설치되지 않아 간소화 UI로 실행됩니다.\n"
            "전체 컨트롤 패널을 보려면 아래 중 하나를 설치하세요:\n"
            "- pip install customtkinter\n"
        )
        try:
            messagebox.showwarning("UI 제한 모드", message)
        except Exception:
            # GUI 경고 실패 시 콘솔 안내로 대체
            print(message)
    
    def setup_layout(self):
        return super().setup_layout()
    def setup_controls(self):
        return super().setup_controls()
    def setup_canvas(self):
        return super().setup_canvas()
    def _log(self, message: str):
        """내부 로그에 적재하고, 옵션에 따라 콘솔에도 출력"""
        try:
            self.internal_logs.append(str(message))
            # 메모리 보호: 로그가 너무 커지지 않도록 제한
            if len(self.internal_logs) > 10000:
                self.internal_logs = self.internal_logs[-5000:]
            if getattr(self, 'enable_console_log', False):
                print(message)
        except Exception:
            # 로깅 실패는 무시
            pass

    def setup_manual_parameters_toggle(self):
        return super().setup_manual_parameters_toggle()
    def setup_manual_parameter_controls(self):
        return super().setup_manual_parameter_controls()
    def toggle_manual_parameters(self):
        return super().toggle_manual_parameters()
    def on_parameter_change(self, value):
        return super().on_parameter_change(value)
    def on_preview_toggle(self):
        return super().on_preview_toggle()
    def update_manual_preview(self, filter_only: bool = False):
        return super().update_manual_preview(filter_only)
    
    def _get_seed_signature(self) -> str:
        """현재 시드 상태의 고유 시그니처 생성"""
        try:
            leaf_seeds = tuple(sorted(self.seed_manager.seeds.get("leaf", [])))
            scale_seeds = tuple(sorted(self.seed_manager.seeds.get("scale", [])))
            bg_seeds = tuple(sorted(self.seed_manager.seeds.get("background", [])))
            method = self._get_segmentation_method()
            return f"leaf:{leaf_seeds}_scale:{scale_seeds}_bg:{bg_seeds}_method:{method}"
        except Exception:
            return "empty"
    
    def _invalidate_mask_cache(self):
        """마스크 캐시 무효화 (시드 변경 시 호출)"""
        self._cached_raw_mask = None
        self._cached_scale_mask = None
        self._last_seed_signature = None
        # 리사이징 캐시도 무효화
        self._resize_cache.clear()
        self._cache_version += 1
        print("마스크 캐시 무효화")
    
    def _get_segmentation_method(self) -> str:
        """현재 세그멘테이션 방법 반환 (항상 GrabCut)"""
        return "grabcut"
    
    def on_segmentation_method_change(self, value=None):
        """세그멘테이션 방법 변경 이벤트 (GrabCut 고정)"""
        print("GrabCut 모드 활성화 - OpenCV GrabCut 사용")
        self._invalidate_mask_cache()
    
    def toggle_object_deletion(self):
        """객체 삭제 기능 토글"""
        if hasattr(self, 'object_deletion_enabled'):
            self._object_deletion_enabled = self.object_deletion_enabled.get()
            status = "활성" if self._object_deletion_enabled else "비활성"
            print(f"객체 삭제 기능 {status}화")

    def apply_scale_area_setting(self):
        """UI 입력값으로 Scale 면적(cm^2) 설정 적용"""
        try:
            val_str = self.scale_area_var.get() if hasattr(self, 'scale_area_var') else ""
            val = float(val_str)
            if not np.isfinite(val) or val <= 0:
                raise ValueError("Scale area must be positive")
            self.settings["scale_area_cm2"] = float(val)
            messagebox.showinfo("적용 완료", f"Scale 면적이 {val:.2f} cm²로 설정되었습니다.\n내보내기 시 반영됩니다.")
            self._safe_refocus()
        except Exception:
            try:
                messagebox.showerror("오류", "유효한 숫자를 입력하세요. (예: 4 또는 3.5)")
                self._safe_refocus()
            except Exception:
                pass

    def apply_min_object_area_setting(self):
        """최소 객체 면적(px) 설정 적용 + 마지막 분석 재탐색"""
        try:
            val_str = self.min_object_area_var.get() if hasattr(self, "min_object_area_var") else ""
            val = int(str(val_str).strip())
            if val <= 0:
                raise ValueError("min_object_area must be positive")
        except Exception:
            try:
                messagebox.showerror("오류", "유효한 양의 정수를 입력하세요. (예: 500)")
                self._safe_refocus()
            except Exception:
                pass
            return

        self.settings["min_object_area"] = int(val)
        # 기본 분석 필터에도 동일 값 적용
        try:
            self.easy_params["min_component"] = int(val)
        except Exception:
            pass
        # 수동 미리보기 필터에도 동일 값 적용
        try:
            self.manual_settings["min_area"] = int(val)
        except Exception:
            pass

        self.rerun_last_analysis()

    def _parse_inference_resize_divisor(self) -> float:
        """리사이즈 배율 파싱 (1 이상)."""
        val_str = ""
        if hasattr(self, "inference_resize_var"):
            val_str = self.inference_resize_var.get()
        if not val_str:
            val_str = str(self.settings.get("inference_resize_divisor", 1))
        try:
            val = float(str(val_str).strip())
        except Exception:
            val = 1.0
        if not np.isfinite(val) or val <= 0:
            val = 1.0
        if val < 1.0:
            val = 1.0
        return float(val)

    def apply_inference_resize_setting(self, silent: bool = False):
        """리사이즈 배율 적용 + 이미지 재구성."""
        val = self._parse_inference_resize_divisor()
        if hasattr(self, "inference_resize_var"):
            try:
                self.inference_resize_var.set(str(val if val % 1 != 0 else int(val)))
            except Exception:
                pass
        if val < 1.0:
            if not silent:
                try:
                    messagebox.showerror("오류", "리사이즈 배율은 1 이상의 숫자여야 합니다.")
                    self._safe_refocus()
                except Exception:
                    pass
            return
        self.settings["inference_resize_divisor"] = float(val)
        self._apply_inference_resize(val)

    def _ensure_inference_resize_applied(self):
        """분석 실행 전 리사이즈 상태 보장."""
        val = self._parse_inference_resize_divisor()
        if self.original_image_full is None and self.original_image is not None:
            self.original_image_full = self.original_image.copy()
        if self.original_image_full is None:
            return
        full_h, full_w = self.original_image_full.shape[:2]
        new_w = max(1, int(round(full_w / val)))
        new_h = max(1, int(round(full_h / val)))
        if (
            self.original_image is None
            or self.original_image.shape[:2] != (new_h, new_w)
            or abs(float(val) - float(self._current_resize_divisor)) > 1e-6
        ):
            self._apply_inference_resize(val)

    def _rescale_seed_points(self, scale_x: float, scale_y: float, new_w: int, new_h: int):
        """시드 좌표를 새 크기에 맞게 스케일."""
        try:
            for cls_name, seeds in self.seed_manager.seeds.items():
                if not seeds:
                    continue
                new_seeds = []
                for (x, y) in seeds:
                    nx = int(round(x * scale_x))
                    ny = int(round(y * scale_y))
                    nx = max(0, min(new_w - 1, nx))
                    ny = max(0, min(new_h - 1, ny))
                    new_seeds.append((nx, ny))
                self.seed_manager.seeds[cls_name] = new_seeds
        except Exception:
            pass

    def _apply_inference_resize(self, divisor: float):
        """현재 리사이즈 배율로 이미지/캐시 갱신."""
        if self.original_image_full is None:
            if self.original_image is None:
                return
            self.original_image_full = self.original_image.copy()
        full_h, full_w = self.original_image_full.shape[:2]
        new_w = max(1, int(round(full_w / divisor)))
        new_h = max(1, int(round(full_h / divisor)))

        old_h, old_w = None, None
        if self.original_image is not None:
            old_h, old_w = self.original_image.shape[:2]

        if old_h == new_h and old_w == new_w and abs(float(divisor) - float(self._current_resize_divisor)) <= 1e-6:
            return

        interp = cv2.INTER_AREA if divisor > 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(self.original_image_full, (new_w, new_h), interpolation=interp)
        self.original_image = resized
        self.working_image = resized
        self._current_resize_divisor = float(divisor)

        # 시드 스케일 조정
        if old_w and old_h:
            scale_x = new_w / float(old_w)
            scale_y = new_h / float(old_h)
            self._rescale_seed_points(scale_x, scale_y, new_w, new_h)

        # 캐시 및 분석 상태 초기화 (시드 유지)
        self._invalidate_mask_cache()
        self.current_masks = {"leaf": None, "scale": None, "background": None}
        self.analysis_results = None
        self._deleted_objects.clear()
        self._deleted_scale_objects.clear()
        self._current_instance_labels = None
        self._current_scale_labels = None
        self.superpixel_labels = None
        self.superpixel_count = 0
        self.seed_segment_ids = {"leaf": set(), "scale": set(), "background": set()}
        self.split_mode_enabled = False
        self.merge_mode_enabled = False
        self.delete_mode_enabled = False
        self.split_mode_points = []
        self.merge_selected = set()
        self.delete_selected = set()

        self.update_display_image()

    def rerun_last_analysis(self):
        """방금 수행한 분석을 동일 파라미터로 재실행"""
        try:
            if self.original_image is None:
                messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
                self._safe_refocus()
                return
            kind = getattr(self, "_last_analysis_kind", None)
            if kind is None:
                messagebox.showinfo("안내", "재탐색할 분석 기록이 없습니다.")
                self._safe_refocus()
                return
            if kind == "basic":
                self.basic_analyze()
            elif kind == "advanced":
                self.analyze_image(forced=True)
            elif kind == "sam3":
                self.mixed_analyze_sam3()
            else:
                messagebox.showinfo("안내", "재탐색할 분석 기록이 없습니다.")
                self._safe_refocus()
        except Exception:
            try:
                messagebox.showerror("오류", "재탐색 중 오류가 발생했습니다.")
                self._safe_refocus()
            except Exception:
                pass

    # === YOLO Export Helpers ===
    
    def _deactivate_delete_mode(self):
        return super()._deactivate_delete_mode()

    def export_yolo_obb(self):
        return super().export_yolo_obb()

    def export_yolo_seg(self):
        return super().export_yolo_seg()

    def export_coco_seg(self):
        return super().export_coco_seg()
    
    def on_object_delete_click(self, event):
        return super().on_object_delete_click(event)

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
            # 팔레트 (RGB) - 시각적으로 구분 쉬운 색상들
            if palette_type == 'scale':
                # 고대비 28색 팔레트 (따뜻/차가운 색 균형)
                palette = [
                    (255,0,0),(255,128,0),(255,0,128),(255,64,0),(255,0,64),(200,0,200),(255,0,255),
                    (255,128,128),(200,80,0),(255,64,160),(255,200,0),(255,160,0),(255,96,0),(255,64,64),
                    (0,255,255),(0,200,255),(0,160,255),(64,224,208),(0,128,255),(0,96,255),(0,64,255),
                    (128,128,255),(96,96,255),(64,64,255),(0,0,255),(32,160,255),(0,180,220),(0,140,200)
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

    def _overlay_by_labels(self, image_rgb: np.ndarray, labels_map: np.ndarray, include_ids: list[int] | set[int], palette_type: str = 'leaf', alpha: float = 0.35, contour_thickness: int = 1, highlight_ids: set[int] | None = None) -> np.ndarray:
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
                    (255,0,0),(255,128,0),(255,0,128),(255,64,0),(255,0,64),(200,0,200),(255,0,255),
                    (255,128,128),(200,80,0),(255,64,160),(255,200,0),(255,160,0),(255,96,0),(255,64,64),
                    (0,255,255),(0,200,255),(0,160,255),(64,224,208),(0,128,255),(0,96,255),(0,64,255),
                    (128,128,255),(96,96,255),(64,64,255),(0,0,255),(32,160,255),(0,180,220),(0,140,200)
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

    def create_preview_overlay(self, leaf_mask, scale_mask=None, stats: dict | None = None):
        return super().create_preview_overlay(leaf_mask, scale_mask, stats)
    def show_preview_image(self, preview_image):
        return super().show_preview_image(preview_image)
    def manual_analyze(self):
        """세그멘테이션 기반 수동 분석 실행 (캐시 활용)"""
        if self.original_image is None:
            messagebox.showerror("오류", "먼저 이미지를 로드해주세요.")
            self._safe_refocus()
            return
        try:
            # 캐시 활용으로 빠른 분석
            current_signature = self._get_seed_signature()
            if (self._cached_raw_mask is None or 
                current_signature != self._last_seed_signature):
                print("시드 변경 감지 - 마스크 재생성")
                self._cached_raw_mask = self.generate_leaf_mask()
                self._last_seed_signature = current_signature
            else:
                print("캐시된 마스크 사용 - 세그멘테이션 스킵")
            
            # 현재 파라미터로 필터링 적용 (이미 min_area 필터링 완료)
            manual_mask = self._apply_size_filter(self._cached_raw_mask)
            
            # Leaf 인스턴스 라벨맵 생성 (객체 삭제 기능을 위해)
            num_labels, instance_labels = cv2.connectedComponents(
                manual_mask.astype(np.uint8), connectivity=8
            )
            self._current_instance_labels = instance_labels
            print(f"   → Leaf 인스턴스 라벨맵 생성: {num_labels - 1}개 객체")
            
            # Scale 라벨맵 생성 (Scale 시드가 있는 경우)
            if self._cached_scale_mask is not None and np.sum(self._cached_scale_mask) > 0:
                scale_num_labels, scale_labels = cv2.connectedComponents(
                    self._cached_scale_mask.astype(np.uint8), connectivity=8
                )
                self._current_scale_labels = scale_labels
                print(f"   → Scale 개별 객체 라벨맵 생성: {scale_num_labels - 1}개 객체")
            else:
                self._current_scale_labels = None
                print("   → Scale 객체 없음 - 라벨맵 생성 스킵")
            
            contours, _ = cv2.findContours(
                manual_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            filtered_objects = []
            total_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                # 이미 필터링된 마스크이므로 추가 면적 체크 불필요
                obj_data = MorphologicalAnalyzer.analyze_contour(contour)
                filtered_objects.append(obj_data)
                total_area += area
            if filtered_objects:
                result_msg = f"수동 분석 결과:\n\n"
                result_msg += f"탐지된 잎 개수: {len(filtered_objects)}개\n"
                result_msg += f"총 면적: {total_area:.0f} 픽셀\n"
                result_msg += f"평균 면적: {total_area/len(filtered_objects):.0f} 픽셀/개\n\n"
                sorted_objects = sorted(filtered_objects, key=lambda x: x['area'], reverse=True)
                for i, obj in enumerate(sorted_objects[:5]):
                    result_msg += f"잎 {i+1}: {obj['area']:.0f}픽셀 (L:{obj['length']:.1f}, W:{obj['width']:.1f})\n"
                if len(sorted_objects) > 5:
                    result_msg += f"... 외 {len(sorted_objects)-5}개"
                messagebox.showinfo("수동 분석 결과", result_msg)
                self._safe_refocus()
                result_overlay = self.create_result_overlay(manual_mask, filtered_objects)
                self.show_preview_image(result_overlay)
            else:
                min_area = self.manual_settings.get("min_area", 200)
                messagebox.showwarning(
                    "분석 결과",
                                     f"설정된 최소 면적({min_area} 픽셀) 이상의 객체를 찾을 수 없습니다.\n"
                    "파라미터를 조정해보세요."
                )
                self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"수동 분석 중 오류가 발생했습니다:\n{e}")
            self._safe_refocus()
    
    def create_result_overlay(self, mask, objects):
        """분석 결과 오버레이 생성"""
        if not hasattr(self, 'working_image') or self.working_image is None:
            return self.original_image
            
        result_img = self.working_image.copy()
        
        # 마스크 오버레이
        result_img[mask] = result_img[mask] * 0.7 + np.array([0, 255, 0]) * 0.3
        
        # 객체 경계 및 라벨 표시
        for i, obj in enumerate(objects[:10]):  # 상위 10개만 표시
            if 'contour' in obj:
                # 경계 그리기
                cv2.drawContours(result_img, [obj['contour']], -1, (255, 255, 0), 2)
                
                # 중심점에 번호 표시
                if 'center' in obj:
                    center = tuple(map(int, obj['center']))
                    cv2.putText(result_img, str(i+1), center, 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result_img
    
    def load_image(self):
        """이미지 로드"""
        # print("🚨 load_image 함수가 호출되었습니다!")  # 디버그용
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        # 대화상자 후 포커스 관리
        self._safe_refocus()
        
        if not file_path:
            return
        
        try:
            # 이미지 로드
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.original_image_full = self.original_image.copy()
            self.current_image_path = file_path
            
            working_image = self.original_image
            
            # GrabCut에서는 RGB만 필요
            self.hsv_image = None
            self.lab_image = None

            # SuperPixel 초기화 (on-demand 생성으로 변경)
            self.superpixel_labels = None
            self.superpixel_count = 0
            self.seed_segment_ids = {"leaf": set(), "scale": set(), "background": set()}
            print("SuperPixel 시스템 초기화 완료 (on-demand 생성)")
            
            # 작업용 이미지 저장
            self.working_image = self.original_image
            
            # 새 이미지 로드 시 모든 캐시 데이터 초기화
            self.reset_all_cache()
            
            # 마스크 캐시 초기화
            self._cached_raw_mask = None
            self._cached_scale_mask = None
            self._last_seed_signature = None
            
            # 리사이즈 설정 적용
            self._apply_inference_resize(self._parse_inference_resize_divisor())
            
            # 표시용 이미지 준비
            self.update_display_image()
            
            status_msg = f"이미지를 성공적으로 로드했습니다.\n크기: {self.original_image.shape[:2]}"
            if self.original_image_full is not None:
                full_h, full_w = self.original_image_full.shape[:2]
                if (full_h, full_w) != self.original_image.shape[:2]:
                    status_msg += f"\n리사이즈 적용: {full_w}x{full_h} → {self.original_image.shape[1]}x{self.original_image.shape[0]}"
            
            status_msg += "\n\n이전 시드가 초기화되었습니다."
            messagebox.showinfo("성공", status_msg)
            self._safe_refocus()  # messagebox 후 포커스 관리 (Enter 키 중복 방지)
            
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 중 오류가 발생했습니다:\n{e}")
            self._safe_refocus()  # 오류 메시지 후에도 포커스 관리
    
    def reset_all_cache(self):
        """새 이미지 로드 시 모든 캐시 데이터 초기화"""
        print("모든 캐시 데이터를 초기화합니다...")
        
        # 최적화: 리사이징 캐시 초기화
        self._resize_cache.clear()
        self._cache_version += 1
        
        # 시드 데이터 초기화
        if hasattr(self, 'seed_manager') and self.seed_manager:
            self.seed_manager.seeds = {"leaf": [], "scale": [], "background": []}
            print("시드 데이터 초기화 완료")
        
        # 슈퍼픽셀 세그먼트 ID 초기화
        if hasattr(self, 'seed_segment_ids'):
            self.seed_segment_ids = {"leaf": set(), "scale": set(), "background": set()}
            print("슈퍼픽셀 세그먼트 ID 초기화 완료")
        
        # GrabCut로 전환: 색상 모델 관련 캐시 제거
        
        # 분석 결과 초기화
        if hasattr(self, 'current_masks'):
            self.current_masks = {"leaf": None, "scale": None, "background": None}
        if hasattr(self, 'analysis_results'):
            self.analysis_results = None
            print("분석 결과 초기화 완료")
        
        # 수동 미리보기 관련 초기화
        if hasattr(self, 'manual_params_visible') and self.manual_params_visible:
            print("📋 수동 파라미터 패널이 열려있어서 미리보기를 클리어합니다")
        
        # 마스크 캐시 초기화
        self._cached_raw_mask = None
        self._cached_scale_mask = None
        self._last_seed_signature = None
        print("마스크 캐시 초기화 완료")
        
        # 객체 삭제 정보 초기화
        self._deleted_objects.clear()
        self._deleted_scale_objects.clear()
        self._current_instance_labels = None
        self._current_scale_labels = None
        print("객체 삭제 정보 초기화 완료")

        # 분리 모드 상태 초기화
        self.split_mode_enabled = False
        self.split_mode_points = []
        self.split_selected_object = None
        self._split_snapshot = None
        
        print("전체 캐시 초기화 완료!")
    
    def update_display_image(self):
        return super().update_display_image()
    def add_seed_markers(self, image):
        return super().add_seed_markers(image)
    def _update_seed_snapshot(self, seed_class: str):
        try:
            self._seed_snapshots[seed_class] = list(self.seed_manager.seeds.get(seed_class, []))
        except Exception:
            pass

    def _is_seed_changed(self, seed_class: str) -> bool:
        try:
            return list(self.seed_manager.seeds.get(seed_class, [])) != self._seed_snapshots.get(seed_class, [])
        except Exception:
            return True

    def _mark_dirty_due_to_seed_change(self, seed_class: str):
        try:
            self._model_dirty[seed_class] = True
        except Exception:
            pass

    # GrabCut 전환으로 모델 보장 로직 제거됨
    
    def on_canvas_click(self, event):
        return super().on_canvas_click(event)
    def on_canvas_right_click(self, event):
        return super().on_canvas_right_click(event)
    def clear_current_seeds(self):
        return super().clear_current_seeds()
    def undo_last_seed(self):
        return super().undo_last_seed()
    def analyze_image(self, forced: bool = False):
        self._ensure_inference_resize_applied()
        self._last_analysis_kind = "advanced"
        return super().analyze_image(forced)
    def apply_morphology(self, mask):
        """형태학적 후처리 (Area Opening + Closing)"""
        # 1) 면적 기준 제거로 경계 blob 필터링
        min_blob_area = int(self.settings.get("min_blob_area", 120))
        cleaned = area_opening(mask.astype(bool), area_threshold=min_blob_area, connectivity=1)
        
        # 2) 작은 구멍 메우기 (Closing) - 최적화: 커널 캐싱
        kernel_size = self.settings["morphology_kernel_size"]
        kernel = self._get_morph_kernel(kernel_size)
        final_mask = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return final_mask.astype(bool)
    
    def preview_analysis(self):
        return super().preview_analysis()
    def create_analysis_overlay(self, leaf_mask, scale_mask=None, stats: dict | None = None):
        return super().create_analysis_overlay(leaf_mask, scale_mask, stats)
    def show_preview_overlay(self, preview_image):
        return super().show_preview_overlay(preview_image)
    def show_analysis_results(self):
        return super().show_analysis_results()
    def show_result_overlay(self, stats: dict | None = None):
        return super().show_result_overlay(stats)
    def show_temporary_overlay(self, overlay_image):
        return super().show_temporary_overlay(overlay_image)
    def batch_process(self):
        """배치 처리"""
        messagebox.showinfo("정보", "배치 처리 기능은 구현 중입니다.")
        self._safe_refocus()  # messagebox 후 포커스 관리
    
    def export_csv(self):
        return super().export_csv()

    def export_json(self):
        return super().export_json()

    def basic_analyze(self):
        self._ensure_inference_resize_applied()
        self._last_analysis_kind = "basic"
        return super().basic_analyze()

    def mixed_analyze_sam3(self):
        self._ensure_inference_resize_applied()
        self._last_analysis_kind = "sam3"
        return super().mixed_analyze_sam3()
    def run(self):
        """애플리케이션 실행"""
        self.root.mainloop()
