#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Leaf Area Analyzer - Main Application Class
Leaf/Scale 분석 + SAM3 후보정
"""

import os
import hashlib
import csv
import json
from urllib.parse import urlparse, unquote
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
import time

# Core 모듈
from .core.seed_manager import SeedManager

# Processing 모듈
from .processing.image_processor import ImageProcessor
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

class AdvancedLeafAnalyzer(GUISetup, EventHandlers, LeafAnalyzer, ImageProcessor, OverlayManager, DataExporter, ExportUtils, ParameterEstimator, ObjectOperations):
    """고급 잎 분석기 메인 클래스"""
    
    def __init__(self):
        # 형태학 기본값
        self.manual_settings = {
            "min_area": 1000,
            "morph_kernel": 5,
        }
        
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
            # 오버레이 스타일(글꼴/윤곽 두께) 통일 설정
            "overlay_font_scale": 0.45,
            "overlay_font_thickness": 1,
            "overlay_contour_thickness": 1,
            # Export용 스케일 면적(cm^2)
            "scale_area_cm2": 4.0
        }
        
        self.seed_manager = SeedManager()
        # 후보정 시드용 background 버킷 포함 보장
        if not isinstance(getattr(self.seed_manager, "seeds", None), dict):
            self.seed_manager.seeds = {}
        for cls_name in ("leaf", "scale", "background"):
            if cls_name not in self.seed_manager.seeds or not isinstance(self.seed_manager.seeds.get(cls_name), list):
                self.seed_manager.seeds[cls_name] = []
        # 고급 색상 유틸 미사용
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
        # 파일 탐색기 마지막 위치 기억
        self._ui_prefs_path = str(Path(__file__).resolve().parents[1] / ".leaf_analyzer_ui_prefs.json")
        self.last_browse_dir: str = str(Path.cwd())
        self._load_ui_prefs()
        
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
        # 배치 리뷰 세션 상태
        self.batch_review_items: List[Dict[str, Any]] = []
        self.batch_review_index: int = -1
        self.batch_review_active: bool = False
        self.batch_review_output_root: Optional[str] = None
        self._batch_review_last_saved_index: int = -1

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
        # 전체 바이트 해시 사용: 희소 마스크(스케일)에서 샘플 해시 충돌 방지
        mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
        mask_hash = hashlib.blake2b(mask_u8.tobytes(), digest_size=8).hexdigest()
        nz_count = int(np.count_nonzero(mask_u8))
        base_id = f"{mask_hash}_{h}_{w}_{nz_count}"
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

    def _load_ui_prefs(self):
        """UI 선호 설정(마지막 탐색 폴더) 로드."""
        try:
            p = Path(self._ui_prefs_path)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            last_dir = str(data.get("last_browse_dir", "")).strip()
            if last_dir and Path(last_dir).is_dir():
                self.last_browse_dir = str(Path(last_dir).resolve())
        except Exception:
            pass

    def _save_ui_prefs(self):
        """UI 선호 설정 저장."""
        try:
            p = Path(self._ui_prefs_path)
            payload = {"last_browse_dir": str(self.last_browse_dir)}
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _get_initial_browse_dir(self) -> str:
        """파일/폴더 다이얼로그 initialdir 반환."""
        try:
            d = Path(self.last_browse_dir)
            if d.is_dir():
                return str(d)
        except Exception:
            pass
        try:
            return str(Path.cwd())
        except Exception:
            return "."

    def _set_last_browse_dir(self, path: str):
        """마지막 탐색 폴더 갱신 및 저장."""
        try:
            p = Path(path).expanduser().resolve()
            if p.is_file():
                p = p.parent
            if p.is_dir():
                self.last_browse_dir = str(p)
                self._save_ui_prefs()
        except Exception:
            pass
    
    
    def setup_gui(self):
        super().setup_gui()
        try:
            self._setup_drag_and_drop()
        except Exception:
            pass
        return None

    def _normalize_dropped_path(self, raw_path: str) -> str:
        """드롭 이벤트 문자열을 로컬 파일 경로로 정규화."""
        p = str(raw_path or "").strip()
        if not p:
            return ""
        if p.startswith("{") and p.endswith("}"):
            p = p[1:-1]
        if p.lower().startswith("file://"):
            try:
                parsed = urlparse(p)
                path = unquote(parsed.path or "")
                if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
                    path = path[1:]
                p = path
            except Exception:
                pass
        return p

    def _is_supported_single_image_file(self, file_path: str) -> bool:
        """단일 이미지 열기 지원 확장자 여부."""
        try:
            ext = str(Path(file_path).suffix).lower()
        except Exception:
            return False
        return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def _handle_drop_data(self, data: str):
        """드롭 데이터 문자열 처리: 첫 번째 유효 이미지 파일을 로드."""
        raw = str(data or "").strip()
        if not raw:
            return "break"
        try:
            candidates = list(self.root.tk.splitlist(raw))
        except Exception:
            candidates = [raw]

        normalized = [self._normalize_dropped_path(p) for p in candidates]
        for p in normalized:
            if not p:
                continue
            if os.path.isfile(p) and self._is_supported_single_image_file(p):
                self._set_last_browse_dir(p)
                self._load_image_from_path(p, show_success_message=False)
                return "break"

        try:
            messagebox.showwarning(
                "드래그 앤 드롭",
                "지원되는 이미지 파일(.jpg, .jpeg, .png, .bmp, .tiff, .tif)을 드롭해주세요."
            )
            self._safe_refocus()
        except Exception:
            pass
        return "break"

    def _setup_drag_and_drop(self):
        """캔버스/윈도우 파일 드래그앤드랍 활성화 (가능한 환경에서만)."""
        if not hasattr(self, "root") or self.root is None:
            return
        try:
            try:
                self.root.tk.call("package", "require", "tkdnd")
            except Exception:
                from tkinterdnd2 import TkinterDnD
                TkinterDnD._require(self.root)
        except Exception:
            print("파일 드래그앤드랍 비활성화: tkdnd/tkinterdnd2를 찾을 수 없습니다.")
            return

        callback = self.root.register(self._handle_drop_data)
        bind_script = f"{callback} %D"
        targets = []
        for w in (getattr(self, "canvas", None), getattr(self, "right_frame", None), self.root):
            if w is None:
                continue
            if w in targets:
                continue
            targets.append(w)

        registered = False
        for widget in targets:
            try:
                self.root.tk.call("tkdnd::drop_target", "register", widget._w, "DND_Files")
                self.root.tk.call("bind", widget._w, "<<Drop>>", bind_script)
                registered = True
            except Exception:
                continue
        if registered:
            print("파일 드래그앤드랍 활성화됨")
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

    def _invalidate_mask_cache(self):
        """리사이즈/표시 캐시 무효화."""
        self._resize_cache.clear()
        self._cache_version += 1
    
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
                        if palette_type == 'scale':
                            overlay = self._draw_dashed_contours(
                                overlay, cnts, outline_color, contour_thickness
                            )
                        else:
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
                        if palette_type == 'scale':
                            img = self._draw_dashed_contours(
                                img, cnts, outline_color, int(max(1, thick))
                            )
                        else:
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
    
    def _load_image_from_path(self, file_path: str, show_success_message: bool = True) -> bool:
        """파일 경로로 이미지 로드."""
        if not file_path:
            return False
        if not self._is_supported_single_image_file(file_path):
            messagebox.showerror("오류", "지원되지 않는 이미지 형식입니다.")
            self._safe_refocus()
            return False
        try:
            # 이미지 로드
            self.original_image = self._imread_unicode(str(file_path), cv2.IMREAD_COLOR)
            if self.original_image is None:
                raise ValueError("이미지를 읽을 수 없습니다.")
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.original_image_full = self.original_image.copy()
            self.current_image_path = file_path

            # RGB 기반 분석이므로 HSV/LAB 캐시는 비움
            self.hsv_image = None
            self.lab_image = None
            
            # 작업용 이미지 저장
            self.working_image = self.original_image
            
            # 새 이미지 로드 시 모든 캐시 데이터 초기화
            self.reset_all_cache()
            
            # 리사이즈 설정 적용
            self._apply_inference_resize(self._parse_inference_resize_divisor())
            
            # 표시용 이미지 준비
            self.update_display_image()
            
            if show_success_message:
                status_msg = f"이미지를 성공적으로 로드했습니다.\n크기: {self.original_image.shape[:2]}"
                if self.original_image_full is not None:
                    full_h, full_w = self.original_image_full.shape[:2]
                    if (full_h, full_w) != self.original_image.shape[:2]:
                        status_msg += f"\n리사이즈 적용: {full_w}x{full_h} → {self.original_image.shape[1]}x{self.original_image.shape[0]}"
                status_msg += "\n\n이전 시드가 초기화되었습니다."
                messagebox.showinfo("성공", status_msg)
                self._safe_refocus()  # messagebox 후 포커스 관리 (Enter 키 중복 방지)
            return True
            
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 중 오류가 발생했습니다:\n{e}")
            self._safe_refocus()  # 오류 메시지 후에도 포커스 관리
            return False

    def load_image(self):
        """이미지 로드"""
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")],
            initialdir=self._get_initial_browse_dir()
        )

        # 대화상자 후 포커스 관리
        self._safe_refocus()
        
        if not file_path:
            return
        self._set_last_browse_dir(file_path)
        self._load_image_from_path(file_path, show_success_message=True)
    
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
        
        # 분석 결과 초기화
        if hasattr(self, 'current_masks'):
            self.current_masks = {"leaf": None, "scale": None, "background": None}
        if hasattr(self, 'analysis_results'):
            self.analysis_results = None
            print("분석 결과 초기화 완료")
        
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
    
    def on_canvas_click(self, event):
        return super().on_canvas_click(event)
    def on_canvas_right_click(self, event):
        return super().on_canvas_right_click(event)
    def clear_current_seeds(self):
        return super().clear_current_seeds()
    def undo_last_seed(self):
        return super().undo_last_seed()
    def apply_sam3_seed_correction(self, show_message: bool = True):
        self._ensure_inference_resize_applied()
        return super().apply_sam3_seed_correction(show_message=show_message)
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
    
    def create_analysis_overlay(self, leaf_mask, scale_mask=None, stats: dict | None = None):
        return super().create_analysis_overlay(leaf_mask, scale_mask, stats)
    def show_preview_overlay(self, preview_image):
        return super().show_preview_overlay(preview_image)
    def show_analysis_results(self, show_message: bool = True):
        return super().show_analysis_results(show_message=show_message)
    def show_result_overlay(self, stats: dict | None = None):
        return super().show_result_overlay(stats)
    def show_temporary_overlay(self, overlay_image):
        return super().show_temporary_overlay(overlay_image)

    def _create_export_overlay(self, leaf_mask, scale_mask=None, stats: dict | None = None):
        """원본 해상도 저장용 오버레이 생성 (텍스트 자동 확대)."""
        base = self.working_image if getattr(self, "working_image", None) is not None else self.original_image
        if base is None:
            return self.create_analysis_overlay(leaf_mask, scale_mask, stats)
        h, w = base.shape[:2]
        long_side = float(max(h, w))
        # 1600px 기준으로 텍스트를 점진 확대 (UI 표시에는 영향 없음)
        text_scale_mult = float(np.clip(long_side / 1600.0, 1.0, 3.5))

        old_font_scale = self.settings.get("overlay_font_scale", 0.45)
        old_font_thickness = self.settings.get("overlay_font_thickness", 1)
        try:
            self.settings["overlay_font_scale"] = float(old_font_scale) * text_scale_mult
            self.settings["overlay_font_thickness"] = int(max(1, round(float(old_font_thickness) * text_scale_mult)))
            return self.create_analysis_overlay(leaf_mask, scale_mask, stats)
        finally:
            self.settings["overlay_font_scale"] = old_font_scale
            self.settings["overlay_font_thickness"] = old_font_thickness

    def _set_batch_review_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for attr in ("batch_review_prev_btn", "batch_review_save_btn", "batch_review_next_btn"):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            try:
                btn.configure(state=state)
            except Exception:
                pass

    def _update_batch_review_ui(self):
        total = len(getattr(self, "batch_review_items", []))
        idx = int(getattr(self, "batch_review_index", -1))
        if total <= 0 or idx < 0 or idx >= total:
            if hasattr(self, "batch_review_label") and self.batch_review_label is not None:
                try:
                    self.batch_review_label.configure(text="배치 리뷰: 없음")
                except Exception:
                    pass
            self._set_batch_review_controls_enabled(False)
            return
        item = self.batch_review_items[idx]
        status = str(item.get("status", ""))
        name = str(item.get("image_name", ""))
        text = f"배치 리뷰: {idx+1}/{total} | {name} | {status}"
        if hasattr(self, "batch_review_label") and self.batch_review_label is not None:
            try:
                self.batch_review_label.configure(text=text)
            except Exception:
                pass
        self._set_batch_review_controls_enabled(True)

    def _capture_current_review_snapshot(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "analysis_results": None,
            "instance_labels": None,
            "scale_labels": None,
            "deleted_leaf": set(),
            "deleted_scale": set(),
        }
        try:
            if self.analysis_results is not None:
                snap["analysis_results"] = deepcopy(self.analysis_results)
            if self._current_instance_labels is not None:
                snap["instance_labels"] = self._current_instance_labels.copy()
            if self._current_scale_labels is not None:
                snap["scale_labels"] = self._current_scale_labels.copy()
            snap["deleted_leaf"] = set(getattr(self, "_deleted_objects", set()))
            snap["deleted_scale"] = set(getattr(self, "_deleted_scale_objects", set()))
        except Exception:
            pass
        return snap

    def _batch_review_store_current(self):
        if not getattr(self, "batch_review_active", False):
            return
        idx = int(getattr(self, "batch_review_index", -1))
        items = getattr(self, "batch_review_items", [])
        if idx < 0 or idx >= len(items):
            return
        item = items[idx]
        item["snapshot"] = self._capture_current_review_snapshot()
        item["review_modified"] = True
        # 간단 상태 갱신
        snap = item.get("snapshot", {})
        ar = snap.get("analysis_results", None) if isinstance(snap, dict) else None
        status = "OK"
        reasons = []
        if ar is None:
            status = "NEEDS_REVIEW"
            reasons.append("no_analysis_result")
        else:
            objs = ar.get("objects", []) if isinstance(ar, dict) else []
            if len(objs) <= 0:
                status = "NEEDS_REVIEW"
                reasons.append("no_leaf_objects")
        item["status"] = status
        item["reasons"] = reasons

    def _batch_review_load_index(self, index: int):
        items = getattr(self, "batch_review_items", [])
        if index < 0 or index >= len(items):
            return
        item = items[index]
        image_path = item.get("image_path", "")
        bgr = self._imread_unicode(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("배치 리뷰", f"이미지를 읽을 수 없습니다:\n{image_path}")
            self._safe_refocus()
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.original_image = rgb
        self.original_image_full = rgb.copy()
        self.current_image_path = str(image_path)
        self.working_image = rgb
        self.hsv_image = None
        self.lab_image = None
        self.reset_all_cache()
        # 배치 실행 시점 리사이즈 재적용
        try:
            div = float(item.get("resize_divisor", self.settings.get("inference_resize_divisor", 1.0)))
        except Exception:
            div = float(self.settings.get("inference_resize_divisor", 1.0))
        if div < 1.0:
            div = 1.0
        self.settings["inference_resize_divisor"] = float(div)
        if hasattr(self, "inference_resize_var"):
            try:
                self.inference_resize_var.set(str(div if div % 1 else int(div)))
            except Exception:
                pass
        self._apply_inference_resize(float(div))

        self.seed_manager.seeds = {"leaf": [], "scale": [], "background": []}
        self._deleted_objects = set()
        self._deleted_scale_objects = set()
        self.analysis_results = None
        self._current_instance_labels = None
        self._current_scale_labels = None

        snap = item.get("snapshot", None)
        if isinstance(snap, dict):
            self.analysis_results = deepcopy(snap.get("analysis_results", None))
            inst = snap.get("instance_labels", None)
            scl = snap.get("scale_labels", None)
            self._current_instance_labels = None if inst is None else inst.copy()
            self._current_scale_labels = None if scl is None else scl.copy()
            self._deleted_objects = set(snap.get("deleted_leaf", set()))
            self._deleted_scale_objects = set(snap.get("deleted_scale", set()))

        self.batch_review_index = int(index)
        self.update_display_image()
        if self.analysis_results is not None:
            try:
                stats = self._compute_result_stats()
                self.show_result_overlay(stats)
            except Exception:
                self.show_result_overlay()
        self._update_batch_review_ui()

    def batch_review_prev(self):
        if not getattr(self, "batch_review_active", False):
            return
        idx = int(getattr(self, "batch_review_index", -1))
        if idx <= 0:
            return
        self._batch_review_store_current()
        self._batch_review_load_index(idx - 1)

    def batch_review_next(self):
        if not getattr(self, "batch_review_active", False):
            return
        idx = int(getattr(self, "batch_review_index", -1))
        items = getattr(self, "batch_review_items", [])
        if idx < 0 or idx >= len(items) - 1:
            return
        self._batch_review_store_current()
        self._batch_review_load_index(idx + 1)

    def batch_review_save_current(self):
        if not getattr(self, "batch_review_active", False):
            messagebox.showinfo("배치 리뷰", "활성화된 배치 리뷰 세션이 없습니다.")
            self._safe_refocus()
            return
        self._batch_review_store_current()
        try:
            self._save_current_review_item_to_disk()
        except Exception:
            pass
        self._batch_review_last_saved_index = int(getattr(self, "batch_review_index", -1))
        self._update_batch_review_ui()
        messagebox.showinfo("배치 리뷰", "현재 이미지 수정 결과를 배치 리뷰 세션에 저장했습니다.")
        self._safe_refocus()

    def _save_current_review_item_to_disk(self):
        out_root = getattr(self, "batch_review_output_root", None)
        if not out_root:
            return
        idx = int(getattr(self, "batch_review_index", -1))
        items = getattr(self, "batch_review_items", [])
        if idx < 0 or idx >= len(items):
            return
        item = items[idx]
        image_name = str(item.get("image_name", "image"))
        stem = Path(image_name).stem

        reports_dir = os.path.join(out_root, "reports")
        overlays_dir = os.path.join(out_root, "overlays")
        per_image_dir = os.path.join(out_root, "per_image")
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)

        # summary
        snap = item.get("snapshot", {}) if isinstance(item.get("snapshot", {}), dict) else {}
        ar = snap.get("analysis_results", None)
        status = str(item.get("status", "OK"))
        reasons = item.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        leaf_count = 0
        scale_count = 0
        leaf_area_px = 0
        scale_area_px = 0
        ppcm2 = None
        leaf_rows = []
        scale_rows = []

        if isinstance(ar, dict):
            objs = ar.get("objects", []) or []
            leaf_count = int(len(objs))
            leaf_area_px = int(ar.get("total_leaf_area_pixels", 0) or 0)
            ppcm2 = ar.get("pixels_per_cm2", None)
            for obj in objs:
                cx, cy = obj.get("center", (0.0, 0.0))
                leaf_rows.append({
                    "object_type": "leaf",
                    "object_id": int(obj.get("id", -1)),
                    "area_pixels": float(obj.get("area", 0.0)),
                    "length_pixels": float(obj.get("length", 0.0)),
                    "width_pixels": float(obj.get("width", 0.0)),
                    "perimeter_pixels": float(obj.get("perimeter", 0.0)),
                    "center_x": float(cx),
                    "center_y": float(cy),
                })

            scl = snap.get("scale_labels", None)
            if scl is not None:
                labels = np.asarray(scl)
                for sid in np.unique(labels):
                    sid_i = int(sid)
                    if sid_i <= 0:
                        continue
                    m = (labels == sid_i)
                    a = int(np.sum(m))
                    if a <= 0:
                        continue
                    ys, xs = np.where(m)
                    cx = float(np.mean(xs)) if xs.size > 0 else 0.0
                    cy = float(np.mean(ys)) if ys.size > 0 else 0.0
                    scale_rows.append({
                        "object_type": "scale",
                        "object_id": sid_i,
                        "area_pixels": a,
                        "length_pixels": None,
                        "width_pixels": None,
                        "perimeter_pixels": None,
                        "center_x": cx,
                        "center_y": cy,
                    })
            elif ar.get("scale_mask", None) is not None:
                sm = np.asarray(ar.get("scale_mask")).astype(np.uint8)
                if sm.size > 0 and int(np.sum(sm)) > 0:
                    num_labels, labels = cv2.connectedComponents(sm, connectivity=8)
                    for sid in range(1, int(num_labels)):
                        m = (labels == sid)
                        a = int(np.sum(m))
                        if a <= 0:
                            continue
                        ys, xs = np.where(m)
                        cx = float(np.mean(xs)) if xs.size > 0 else 0.0
                        cy = float(np.mean(ys)) if ys.size > 0 else 0.0
                        scale_rows.append({
                            "object_type": "scale",
                            "object_id": sid,
                            "area_pixels": a,
                            "length_pixels": None,
                            "width_pixels": None,
                            "perimeter_pixels": None,
                            "center_x": cx,
                            "center_y": cy,
                        })
            scale_count = int(len(scale_rows))
            scale_area_px = int(sum(r.get("area_pixels", 0) or 0 for r in scale_rows))

        # CSV
        if os.path.isdir(per_image_dir):
            csv_path = os.path.join(per_image_dir, f"{stem}.csv")
            json_path = os.path.join(per_image_dir, f"{stem}.json")
        else:
            csv_path = os.path.join(reports_dir, f"{stem}_review.csv")
            json_path = os.path.join(reports_dir, f"{stem}_review.json")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            writer.writerow(["image_name", image_name])
            writer.writerow(["status", status])
            writer.writerow(["reasons", "|".join(reasons)])
            writer.writerow(["leaf_count", leaf_count])
            writer.writerow(["scale_count", scale_count])
            writer.writerow(["leaf_area_pixels", leaf_area_px])
            writer.writerow(["scale_area_pixels", scale_area_px])
            writer.writerow(["pixels_per_cm2", ppcm2 if ppcm2 is not None else ""])
            writer.writerow([])
            writer.writerow(["object_type", "object_id", "area_pixels", "length_pixels", "width_pixels", "perimeter_pixels", "center_x", "center_y"])
            for row in leaf_rows + scale_rows:
                writer.writerow([
                    row.get("object_type"),
                    row.get("object_id"),
                    row.get("area_pixels"),
                    row.get("length_pixels", ""),
                    row.get("width_pixels", ""),
                    row.get("perimeter_pixels", ""),
                    row.get("center_x"),
                    row.get("center_y"),
                ])

        # JSON
        payload = {
            "image_name": image_name,
            "status": status,
            "reasons": reasons,
            "leaf_count": leaf_count,
            "scale_count": scale_count,
            "leaf_area_pixels": leaf_area_px,
            "scale_area_pixels": scale_area_px,
            "pixels_per_cm2": ppcm2,
            "objects": leaf_rows + scale_rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)

        # Overlay 이미지 저장
        try:
            if self.analysis_results is not None:
                stats = self._compute_result_stats()
                ov = self._create_export_overlay(
                    self.analysis_results.get("leaf_mask"),
                    self.analysis_results.get("scale_mask"),
                    stats,
                )
                if ov is not None:
                    self._imwrite_unicode(
                        os.path.join(overlays_dir, f"{stem}_overlay.jpg"),
                        cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                    )
        except Exception:
            pass

    def batch_process(self):
        """선택한 폴더의 이미지 파일 전체를 배치 분석하고 통합 결과를 내보낸다."""
        input_dir = filedialog.askdirectory(
            title="배치 처리할 이미지 폴더 선택",
            initialdir=self._get_initial_browse_dir()
        )
        self._safe_refocus()
        if not input_dir:
            return
        self._set_last_browse_dir(input_dir)

        supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_paths = sorted(
            [str(p) for p in Path(input_dir).iterdir() if p.is_file() and p.suffix.lower() in supported_ext]
        )
        if len(image_paths) == 0:
            messagebox.showwarning("배치 처리", "지원되는 이미지 파일이 없습니다.")
            self._safe_refocus()
            return

        analysis_mode = "sam3"
        if hasattr(self, "batch_analysis_mode_var"):
            try:
                analysis_mode = str(self.batch_analysis_mode_var.get()).strip().lower()
            except Exception:
                analysis_mode = "sam3"
        if analysis_mode not in {"basic", "sam3"}:
            analysis_mode = "sam3"

        save_per_image = False
        if hasattr(self, "batch_save_per_image_var"):
            try:
                save_per_image = bool(self.batch_save_per_image_var.get())
            except Exception:
                save_per_image = False

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_root = os.path.join(input_dir, f"batch_output_{timestamp}")
        reports_dir = os.path.join(out_root, "reports")
        overlays_dir = os.path.join(out_root, "overlays")
        per_image_dir = os.path.join(out_root, "per_image")
        yolo_obb_images_dir = os.path.join(out_root, "yolo_obb", "images")
        yolo_obb_labels_dir = os.path.join(out_root, "yolo_obb", "labels")
        yolo_seg_images_dir = os.path.join(out_root, "yolo_seg", "images")
        yolo_seg_labels_dir = os.path.join(out_root, "yolo_seg", "labels")
        coco_images_dir = os.path.join(out_root, "coco", "images")
        for d in [
            reports_dir,
            overlays_dir,
            yolo_obb_images_dir,
            yolo_obb_labels_dir,
            yolo_seg_images_dir,
            yolo_seg_labels_dir,
            coco_images_dir,
        ]:
            os.makedirs(d, exist_ok=True)
        if save_per_image:
            os.makedirs(per_image_dir, exist_ok=True)

        # 클래스 파일(배치 고정)
        with open(os.path.join(out_root, "yolo_obb", "classes.txt"), "w", encoding="utf-8") as f:
            f.write("leaf\nunused\nscale\n")
        with open(os.path.join(out_root, "yolo_seg", "classes.txt"), "w", encoding="utf-8") as f:
            f.write("leaf\nunused\nscale\n")

        use_polygon = False
        if hasattr(self, "use_polygon_format"):
            try:
                use_polygon = bool(self.use_polygon_format.get())
            except Exception:
                use_polygon = False

        coco_data = {
            "info": {"description": "Leaf Area Analyzer Batch Export", "version": "1.0"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "leaf", "supercategory": "plant"},
                {"id": 2, "name": "scale", "supercategory": "measurement"},
            ],
        }
        coco_image_id = 1
        coco_ann_id = 1

        summary_rows = []
        object_rows = []
        json_items = []
        needs_review_rows = []
        review_items = []

        used_stems = set()

        def _unique_stem(stem: str) -> str:
            candidate = stem
            idx = 2
            while candidate in used_stems:
                candidate = f"{stem}_{idx}"
                idx += 1
            used_stems.add(candidate)
            return candidate

        def _collect_scale_masks_current():
            masks = []
            if self._current_scale_labels is not None:
                labels = self._current_scale_labels
                for sid in np.unique(labels):
                    sid_i = int(sid)
                    if sid_i > 0:
                        masks.append((sid_i, labels == sid_i))
                return masks
            if self.analysis_results and self.analysis_results.get("scale_mask") is not None:
                sm = np.asarray(self.analysis_results.get("scale_mask")).astype(np.uint8)
                if sm.size > 0 and int(np.sum(sm)) > 0:
                    num_labels, labels = cv2.connectedComponents(sm, connectivity=8)
                    for sid in range(1, int(num_labels)):
                        masks.append((sid, labels == sid))
            return masks

        def _leaf_mask_for_object(leaf_id: int):
            if self._current_instance_labels is not None:
                m = (self._current_instance_labels == int(leaf_id))
                if np.any(m):
                    return m
            leaf_mask = self.analysis_results.get("leaf_mask") if self.analysis_results else None
            if leaf_mask is not None and np.any(leaf_mask):
                objs = self.analysis_results.get("objects", []) if self.analysis_results else []
                if len(objs) <= 1:
                    return np.asarray(leaf_mask).astype(bool)
            return None

        def _write_per_image_files(stem: str, summary: dict, leaf_objs: list, scale_objs: list):
            if not save_per_image:
                return
            csv_path = os.path.join(per_image_dir, f"{stem}.csv")
            json_path = os.path.join(per_image_dir, f"{stem}.json")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "value"])
                for k in [
                    "image_name",
                    "status",
                    "reasons",
                    "method",
                    "leaf_count",
                    "scale_count",
                    "leaf_area_pixels",
                    "scale_area_pixels",
                    "pixels_per_cm2",
                    "processing_ms",
                ]:
                    writer.writerow([k, summary.get(k, "")])
                writer.writerow([])
                writer.writerow(["object_type", "object_id", "area_pixels", "length_pixels", "width_pixels", "perimeter_pixels", "center_x", "center_y"])
                for row in leaf_objs:
                    writer.writerow([
                        "leaf",
                        row.get("object_id"),
                        row.get("area_pixels"),
                        row.get("length_pixels"),
                        row.get("width_pixels"),
                        row.get("perimeter_pixels"),
                        row.get("center_x"),
                        row.get("center_y"),
                    ])
                for row in scale_objs:
                    writer.writerow([
                        "scale",
                        row.get("object_id"),
                        row.get("area_pixels"),
                        "",
                        "",
                        "",
                        row.get("center_x"),
                        row.get("center_y"),
                    ])
            with open(json_path, "w", encoding="utf-8") as f:
                payload = {"summary": summary, "leaf_objects": leaf_objs, "scale_objects": scale_objs}
                json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)

        total = len(image_paths)
        progress_bar = None
        try:
            try:
                from tqdm.auto import tqdm as _tqdm
                progress_bar = _tqdm(total=total, desc="Batch", unit="img", dynamic_ncols=True, leave=True)
            except Exception:
                progress_bar = None

            for idx, image_path in enumerate(image_paths, start=1):
                image_name = os.path.basename(image_path)
                stem = _unique_stem(Path(image_name).stem)
                short_name = image_name if len(image_name) <= 36 else (image_name[:33] + "...")
                if progress_bar is not None:
                    try:
                        progress_bar.set_postfix_str(short_name, refresh=False)
                    except Exception:
                        pass
                else:
                    print(f"[Batch] ({idx}/{total}) {image_name}")

                t0 = time.perf_counter()
                status = "OK"
                reasons = []
                leaf_rows_this = []
                scale_rows_this = []
                success = False
                leaf_count = 0
                scale_count = 0
                leaf_area_px = 0
                scale_area_px = 0
                pixels_per_cm2 = None

                item_json = {
                    "image_name": image_name,
                    "image_path": image_path,
                    "status": "OK",
                    "reasons": [],
                    "method": analysis_mode,
                    "leaf_count": 0,
                    "scale_count": 0,
                    "leaf_area_pixels": 0,
                    "scale_area_pixels": 0,
                    "pixels_per_cm2": None,
                    "processing_ms": 0.0,
                    "leaf_objects": [],
                    "scale_objects": [],
                }

                try:
                    bgr = self._imread_unicode(image_path, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError("image_read_failed")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    self.original_image = rgb
                    self.original_image_full = rgb.copy()
                    self.current_image_path = image_path
                    self.working_image = rgb
                    self.hsv_image = None
                    self.lab_image = None
                    self.reset_all_cache()
                    self._apply_inference_resize(self._parse_inference_resize_divisor())
    
                    if analysis_mode == "basic":
                        success = bool(self.basic_analyze(show_message=False, show_overlay=False))
                    else:
                        success = bool(self.mixed_analyze_sam3(show_message=False, show_overlay=False))
    
                    if (not success) or (self.analysis_results is None):
                        status = "NEEDS_REVIEW"
                        reasons.append(getattr(self, "_last_analysis_error", None) or "analysis_failed")
                    else:
                        leaf_count = int(len(self.analysis_results.get("objects", [])))
                        leaf_area_px = int(self.analysis_results.get("total_leaf_area_pixels", 0) or 0)
                        scale_mask_now = self.analysis_results.get("scale_mask")
                        scale_area_px = int(np.sum(scale_mask_now)) if scale_mask_now is not None else 0
                        pixels_per_cm2 = self.analysis_results.get("pixels_per_cm2", None)
    
                        scale_masks = _collect_scale_masks_current()
                        scale_count = len(scale_masks)
    
                        if leaf_count <= 0:
                            reasons.append("no_leaf_objects")
                        if leaf_area_px <= 0:
                            reasons.append("zero_leaf_area")
                        if reasons:
                            status = "NEEDS_REVIEW"
    
                        try:
                            stats = self._compute_result_stats()
                            overlay = self._create_export_overlay(
                                self.analysis_results.get("leaf_mask"),
                                self.analysis_results.get("scale_mask"),
                                stats,
                            )
                            if overlay is not None:
                                self._imwrite_unicode(
                                    os.path.join(overlays_dir, f"{stem}_overlay.jpg"),
                                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                                )
                            self.update_display_image()
                            self.show_result_overlay(stats)
                        except Exception:
                            pass
    
                        for obj in self.analysis_results.get("objects", []):
                            cx, cy = obj.get("center", (0.0, 0.0))
                            row = {
                                "image_name": image_name,
                                "object_type": "leaf",
                                "object_id": int(obj.get("id", -1)),
                                "area_pixels": float(obj.get("area", 0.0)),
                                "length_pixels": float(obj.get("length", 0.0)),
                                "width_pixels": float(obj.get("width", 0.0)),
                                "perimeter_pixels": float(obj.get("perimeter", 0.0)),
                                "center_x": float(cx),
                                "center_y": float(cy),
                                "status": status,
                            }
                            leaf_rows_this.append(row)
                            object_rows.append(row)
    
                        for sid, smask in scale_masks:
                            ys, xs = np.where(smask)
                            cx = float(np.mean(xs)) if xs.size > 0 else 0.0
                            cy = float(np.mean(ys)) if ys.size > 0 else 0.0
                            row = {
                                "image_name": image_name,
                                "object_type": "scale",
                                "object_id": int(sid),
                                "area_pixels": int(np.sum(smask)),
                                "length_pixels": None,
                                "width_pixels": None,
                                "perimeter_pixels": None,
                                "center_x": cx,
                                "center_y": cy,
                                "status": status,
                            }
                            scale_rows_this.append(row)
                            object_rows.append(row)
    
                        # YOLO OBB/Seg + COCO는 라벨 품질을 위해 OK 건만 기록
                        if status == "OK":
                            img_export = self.original_image
                            h, w = img_export.shape[:2]
                            yolo_img_name = f"{stem}.jpg"
                            self._imwrite_unicode(
                                os.path.join(yolo_obb_images_dir, yolo_img_name),
                                cv2.cvtColor(img_export, cv2.COLOR_RGB2BGR),
                            )
                            self._imwrite_unicode(
                                os.path.join(yolo_seg_images_dir, yolo_img_name),
                                cv2.cvtColor(img_export, cv2.COLOR_RGB2BGR),
                            )
                            self._imwrite_unicode(
                                os.path.join(coco_images_dir, yolo_img_name),
                                cv2.cvtColor(img_export, cv2.COLOR_RGB2BGR),
                            )
    
                            # YOLO OBB
                            obb_lines = []
                            for obj in self.analysis_results.get("objects", []):
                                if "contour" in obj:
                                    cnt = np.asarray(obj["contour"], dtype=np.float32).reshape(-1, 2)
                                    quad = self._robust_obb_from_points(cnt, w, h)
                                elif "bounding_box" in obj:
                                    bb = np.asarray(obj["bounding_box"], dtype=float).reshape(-1, 2)
                                    if self._touching_border(bb, w, h):
                                        quad = self._aabb(bb)
                                    else:
                                        quad = bb
                                else:
                                    continue
                                obb_lines.append(self._yolo_obb_line(0, quad, w, h))
                            for _, smask in scale_masks:
                                quad = self._scale_obb_from_mask(smask, w, h)
                                if quad is None:
                                    ys, xs = np.where(smask)
                                    if xs.size == 0:
                                        continue
                                    pts = np.column_stack([xs, ys]).astype(np.float32)
                                    quad = self._robust_obb_from_points(pts, w, h)
                                obb_lines.append(self._yolo_obb_line(2, quad, w, h))
                            with open(os.path.join(yolo_obb_labels_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                                for line in obb_lines:
                                    f.write(line + "\n")
    
                            # YOLO Seg
                            seg_lines = []
                            for obj in self.analysis_results.get("objects", []):
                                if "contour" not in obj:
                                    continue
                                cnt = np.asarray(obj["contour"], dtype=np.float32).reshape(-1, 2)
                                if cnt.shape[0] < 3:
                                    continue
                                coords_norm = []
                                for x, y in cnt:
                                    coords_norm.append(float(x) / float(w))
                                    coords_norm.append(float(y) / float(h))
                                seg_lines.append((0, coords_norm))
                            for _, smask in scale_masks:
                                polys = self._mask_to_polygon(smask)
                                if not polys:
                                    continue
                                for poly in polys:
                                    if len(poly) < 6:
                                        continue
                                    coords_norm = []
                                    for k in range(0, len(poly), 2):
                                        coords_norm.append(float(poly[k]) / float(w))
                                        coords_norm.append(float(poly[k + 1]) / float(h))
                                    seg_lines.append((2, coords_norm))
                            with open(os.path.join(yolo_seg_labels_dir, f"{stem}.txt"), "w", encoding="utf-8") as f:
                                for cid, coords in seg_lines:
                                    if len(coords) < 6:
                                        continue
                                    f.write(" ".join([str(cid)] + [f"{v:.6f}" for v in coords]) + "\n")
    
                            # COCO
                            current_image_id = coco_image_id
                            coco_image_id += 1
                            coco_data["images"].append({
                                "id": int(current_image_id),
                                "file_name": yolo_img_name,
                                "width": int(w),
                                "height": int(h),
                            })
                            for obj in self.analysis_results.get("objects", []):
                                oid = int(obj.get("id", -1))
                                if oid <= 0:
                                    continue
                                mask = _leaf_mask_for_object(oid)
                                if mask is None or not np.any(mask):
                                    continue
                                bbox = self._bbox_from_mask(mask)
                                if bbox is None:
                                    continue
                                x0, y0, x1, y1 = bbox
                                if use_polygon:
                                    segm = self._mask_to_polygon(mask)
                                else:
                                    segm = {"size": [int(h), int(w)], "counts": self._rle_counts_from_mask(mask)}
                                coco_data["annotations"].append({
                                    "id": int(coco_ann_id),
                                    "image_id": int(current_image_id),
                                    "category_id": 0,
                                    "segmentation": segm,
                                    "area": int(np.sum(mask)),
                                    "bbox": [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)],
                                    "iscrowd": 0,
                                    "leaf_object_id": int(oid),
                                })
                                coco_ann_id += 1
                            for sid, smask in scale_masks:
                                if not np.any(smask):
                                    continue
                                bbox = self._bbox_from_mask(smask)
                                if bbox is None:
                                    continue
                                x0, y0, x1, y1 = bbox
                                if use_polygon:
                                    segm = self._mask_to_polygon(smask)
                                else:
                                    segm = {"size": [int(h), int(w)], "counts": self._rle_counts_from_mask(smask)}
                                coco_data["annotations"].append({
                                    "id": int(coco_ann_id),
                                    "image_id": int(current_image_id),
                                    "category_id": 2,
                                    "segmentation": segm,
                                    "area": int(np.sum(smask)),
                                    "bbox": [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)],
                                    "iscrowd": 0,
                                    "scale_object_id": int(sid),
                                })
                                coco_ann_id += 1
    
                except Exception as e:
                    status = "NEEDS_REVIEW"
                    reasons.append(f"exception:{e}")

                processing_ms = (time.perf_counter() - t0) * 1000.0
                item_json["status"] = status
                item_json["reasons"] = reasons
                item_json["leaf_count"] = leaf_count
                item_json["scale_count"] = scale_count
                item_json["leaf_area_pixels"] = leaf_area_px
                item_json["scale_area_pixels"] = scale_area_px
                item_json["pixels_per_cm2"] = pixels_per_cm2
                item_json["processing_ms"] = round(processing_ms, 2)
                item_json["leaf_objects"] = leaf_rows_this
                item_json["scale_objects"] = scale_rows_this
                json_items.append(item_json)

                reason_text = "|".join(reasons) if reasons else ""
                summary_row = {
                    "image_name": image_name,
                    "status": status,
                    "reasons": reason_text,
                    "method": analysis_mode,
                    "leaf_count": leaf_count,
                    "scale_count": scale_count,
                    "leaf_area_pixels": leaf_area_px,
                    "scale_area_pixels": scale_area_px,
                    "pixels_per_cm2": pixels_per_cm2 if pixels_per_cm2 is not None else "",
                    "processing_ms": round(processing_ms, 2),
                }
                summary_rows.append(summary_row)
                if status == "NEEDS_REVIEW":
                    needs_review_rows.append(summary_row)

                review_snapshot = {
                    "analysis_results": None,
                    "instance_labels": None,
                    "scale_labels": None,
                    "deleted_leaf": set(),
                    "deleted_scale": set(),
                }
                try:
                    if self.analysis_results is not None:
                        review_snapshot = self._capture_current_review_snapshot()
                except Exception:
                    pass
                review_items.append({
                    "image_name": image_name,
                    "image_path": image_path,
                    "status": status,
                    "reasons": list(reasons),
                    "mode": analysis_mode,
                    "resize_divisor": float(getattr(self, "_current_resize_divisor", self.settings.get("inference_resize_divisor", 1.0))),
                    "snapshot": review_snapshot,
                    "review_modified": False,
                })

                _write_per_image_files(stem, summary_row, leaf_rows_this, scale_rows_this)
                try:
                    self.root.update_idletasks()
                except Exception:
                    pass

                if progress_bar is not None:
                    try:
                        state = "OK" if status == "OK" else "REVIEW"
                        progress_bar.set_postfix_str(f"{state} | {short_name}", refresh=False)
                        progress_bar.update(1)
                    except Exception:
                        pass
        finally:
            if progress_bar is not None:
                try:
                    progress_bar.close()
                except Exception:
                    pass

        # 통합 CSV
        summary_csv_path = os.path.join(reports_dir, "batch_summary.csv")
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_name",
                    "status",
                    "reasons",
                    "method",
                    "leaf_count",
                    "scale_count",
                    "leaf_area_pixels",
                    "scale_area_pixels",
                    "pixels_per_cm2",
                    "processing_ms",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        objects_csv_path = os.path.join(reports_dir, "batch_objects.csv")
        with open(objects_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_name",
                    "object_type",
                    "object_id",
                    "area_pixels",
                    "length_pixels",
                    "width_pixels",
                    "perimeter_pixels",
                    "center_x",
                    "center_y",
                    "status",
                ],
            )
            writer.writeheader()
            for row in object_rows:
                writer.writerow(row)

        if needs_review_rows:
            review_csv_path = os.path.join(reports_dir, "needs_review.csv")
            with open(review_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_name",
                        "status",
                        "reasons",
                        "method",
                        "leaf_count",
                        "scale_count",
                        "leaf_area_pixels",
                        "scale_area_pixels",
                        "pixels_per_cm2",
                        "processing_ms",
                    ],
                )
                writer.writeheader()
                for row in needs_review_rows:
                    writer.writerow(row)

        # 통합 JSON
        batch_json_path = os.path.join(reports_dir, "batch_results.json")
        with open(batch_json_path, "w", encoding="utf-8") as f:
            payload = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_dir": input_dir,
                "analysis_mode": analysis_mode,
                "total_images": len(image_paths),
                "needs_review_count": len(needs_review_rows),
                "save_per_image": save_per_image,
                "items": json_items,
            }
            json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)

        # COCO JSON
        coco_json_path = os.path.join(out_root, "coco", "annotations.json")
        with open(coco_json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(coco_data), f, ensure_ascii=False, indent=2)

        ok_count = len([r for r in summary_rows if r.get("status") == "OK"])
        review_count = len(needs_review_rows)
        done_msg = (
            f"배치 처리 완료\n\n"
            f"총 이미지: {len(image_paths)}\n"
            f"정상(OK): {ok_count}\n"
            f"검토필요(NEEDS_REVIEW): {review_count}\n"
            f"분석 모드: {analysis_mode}\n"
            f"이미지별 CSV/JSON: {'ON' if save_per_image else 'OFF'}\n\n"
            f"배치 리뷰: 이전/다음으로 전체 이미지 탐색 후 '현재 저장' 가능\n"
            f"(저장 위치: 기존 batch_output의 reports/overlays 또는 per_image 덮어쓰기)\n\n"
            f"출력 폴더:\n{out_root}"
        )
        # 배치 리뷰 세션 활성화 (OK/NEEDS_REVIEW 전체 탐색)
        self.batch_review_items = review_items
        self.batch_review_output_root = out_root
        self.batch_review_active = len(review_items) > 0
        self.batch_review_index = 0 if self.batch_review_active else -1
        self._batch_review_last_saved_index = -1
        self._update_batch_review_ui()
        if self.batch_review_active:
            try:
                self._batch_review_load_index(0)
            except Exception:
                pass
        messagebox.showinfo("배치 처리", done_msg)
        self._safe_refocus()
    
    def export_csv(self):
        return super().export_csv()

    def export_json(self):
        return super().export_json()

    def basic_analyze(self, show_message: bool = True, show_overlay: bool = True):
        self._ensure_inference_resize_applied()
        self._last_analysis_kind = "basic"
        return super().basic_analyze(show_message=show_message, show_overlay=show_overlay)

    def mixed_analyze_sam3(self, show_message: bool = True, show_overlay: bool = True):
        self._ensure_inference_resize_applied()
        self._last_analysis_kind = "sam3"
        return super().mixed_analyze_sam3(show_message=show_message, show_overlay=show_overlay)
    def run(self):
        """애플리케이션 실행"""
        self.root.mainloop()
