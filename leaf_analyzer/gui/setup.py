#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Setup for Advanced Leaf Analyzer
GUI 레이아웃 및 위젯 생성
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False

import cv2
import numpy as np
from PIL import Image, ImageTk

class GUISetup:
    """GUI 설정 믹스인 클래스"""

    def setup_gui(self):
        """GUI 초기화"""
        if CTK_AVAILABLE:
            self.root = ctk.CTk()
            self.root.title("Advanced Leaf Area Analyzer")
            self.root.geometry("1400x900")
        else:
            self.root = tk.Tk()
            self.root.title("Advanced Leaf Area Analyzer")
            self.root.geometry("1400x900")
        self._warn_if_ctk_missing()
        
        # Return/Enter 키가 포커스된 버튼을 활성화하지 않도록 바인딩
        # 이 바인딩은 messagebox 후 Enter 키 중복 실행 문제를 방지
        self._block_return_key = False
        # 버튼 클릭 추적 (이전 클릭 동작 반복 실행 방지)
        self._button_action_handlers = {}
        self._widget_action_map = {}
        self._action_buttons = {}
        def _on_return(event):
            if self._block_return_key:
                return "break"  # 이벤트 전파 중단
            return None
        self.root.bind("<Return>", _on_return)
        self.root.bind("<KP_Enter>", _on_return)  # 숫자패드 Enter

        def _register_action_widgets(widget, action_id):
            try:
                self._widget_action_map[widget] = action_id
                for child in widget.winfo_children():
                    _register_action_widgets(child, action_id)
            except Exception:
                pass

        def _resolve_action_from_widget(widget):
            try:
                w = widget
                while w is not None:
                    if w in self._widget_action_map:
                        return self._widget_action_map[w]
                    w = getattr(w, "master", None)
            except Exception:
                pass
            return None

        def _invoke_action(action_id):
            handler = getattr(self, "_button_action_handlers", {}).get(action_id)
            btn = getattr(self, "_action_buttons", {}).get(action_id)
            state = None
            try:
                if btn is not None:
                    state = btn.cget("state")
            except Exception:
                state = None
            if state in ("disabled", "disable"):
                return None
            if handler is None:
                return None
            return handler()

        def _on_root_press(event):
            action_id = _resolve_action_from_widget(getattr(event, "widget", None))
            if action_id:
                _invoke_action(action_id)
                return None

        def _wrap_button_command(action_id, func):
            # 핸들러 등록 (실제 실행은 press에서)
            try:
                self._button_action_handlers[action_id] = func
            except Exception:
                pass
            def _noop():
                return None
            return _noop

        self._wrap_button_command = _wrap_button_command
        def _make_ctk_button(parent, action_id, **kwargs):
            cmd = kwargs.pop("command", None)
            if cmd is not None:
                cmd = self._wrap_button_command(action_id, cmd)
            btn = ctk.CTkButton(parent, command=cmd, **kwargs)
            try:
                self._action_buttons[action_id] = btn
            except Exception:
                pass
            _register_action_widgets(btn, action_id)
            return btn
        self._make_ctk_button = _make_ctk_button
        self.root.bind_all("<ButtonPress-1>", _on_root_press, add="+")

        self.setup_layout()
        self.setup_controls()
        self.setup_canvas()

    def setup_layout(self):
        """레이아웃 설정"""
        # 메인 프레임 분할 (bottom_frame 제거하여 canvas 확장)
        if CTK_AVAILABLE:
            # 스크롤 가능한 좌측 패널 (CTkScrollableFrame 사용)
            self.left_frame = ctk.CTkScrollableFrame(
                self.root, 
                width=300, 
                height=800,  # 최대 높이 설정
                corner_radius=10,
                scrollbar_button_color=("gray70", "gray30"),
                scrollbar_button_hover_color=("gray60", "gray40")
            )
            self.right_frame = ctk.CTkFrame(self.root)
        else:
            # tkinter 버전에서는 기본 Frame 사용 (스크롤 없음)
            self.left_frame = ttk.Frame(self.root, width=300)
            self.right_frame = ttk.Frame(self.root)
        
        # 고정 폭 유지를 위한 설정
        self.left_frame.pack(side="left", fill="y", padx=5, pady=5)
        if CTK_AVAILABLE:
            # CTkScrollableFrame은 pack_propagate 설정이 다름
            pass  # 자동으로 스크롤 관리됨
        else:
            self.left_frame.pack_propagate(False)  # tkinter용 고정 크기
            
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

    def setup_controls(self):
        """컨트롤 패널 설정"""
        if CTK_AVAILABLE:
            # 파일 조작
            file_frame = ctk.CTkFrame(self.left_frame)
            file_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(file_frame, text="파일 조작", font=("Arial", 14, "bold")).pack(pady=5)

            self._make_ctk_button(file_frame, "load_image", text="이미지 열기", command=self.load_image).pack(pady=2, fill="x")
            self._make_ctk_button(file_frame, "batch_process", text="배치 처리", command=self.batch_process).pack(pady=2, fill="x")
            
            # 시드 선택
            seed_frame = ctk.CTkFrame(self.left_frame)
            seed_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(seed_frame, text="시드 선택", font=("Arial", 14, "bold")).pack(pady=5)
            
            self.seed_mode = ctk.StringVar(value="leaf")
            ctk.CTkRadioButton(seed_frame, text="잎 (Leaf)", variable=self.seed_mode, value="leaf").pack(anchor="w")
            ctk.CTkRadioButton(seed_frame, text="스케일 (Scale)", variable=self.seed_mode, value="scale").pack(anchor="w")
            ctk.CTkRadioButton(seed_frame, text="배경 (BG)", variable=self.seed_mode, value="background").pack(anchor="w")
            
            self._make_ctk_button(seed_frame, "clear_current_seeds", text="시드 초기화", command=self.clear_current_seeds).pack(pady=2, fill="x")
            self._make_ctk_button(seed_frame, "undo_last_seed", text="실행 취소 (Undo)", command=self.undo_last_seed).pack(pady=2, fill="x")
            
            # 분석 설정
            analysis_frame = ctk.CTkFrame(self.left_frame)
            analysis_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(analysis_frame, text="분석 설정", font=("Arial", 14, "bold")).pack(pady=5)
            
            self.preview_enabled = ctk.BooleanVar(value=True)
            resize_row = ctk.CTkFrame(analysis_frame)
            resize_row.pack(fill="x", pady=2)
            ctk.CTkLabel(resize_row, text="추론 리사이즈 배율:", width=110, anchor="w").pack(side="left")
            self.inference_resize_var = tk.StringVar(value=str(self.settings.get("inference_resize_divisor", 1)))
            self.inference_resize_entry = ctk.CTkEntry(resize_row, textvariable=self.inference_resize_var, width=80)
            self.inference_resize_entry.pack(side="left", padx=4)
            self._make_ctk_button(resize_row, "apply_inference_resize", text="적용", width=60, command=self.apply_inference_resize_setting).pack(side="left")
            ctk.CTkLabel(resize_row, text="(예: 4 → 1/4)", text_color="gray").pack(side="left", padx=4)
            
            # 기본 분석 파라미터 설정 프레임
            easy_params_frame = ctk.CTkFrame(analysis_frame)
            easy_params_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(easy_params_frame, text="기본 분석 파라미터", font=("Arial", 12, "bold")).pack(pady=2)
            
            # 파라미터 표시 및 설정 버튼
            self.easy_params_label = ctk.CTkLabel(
                easy_params_frame, 
                text=f"G>{self.easy_params['minG']}, G/R>{self.easy_params['ratG']}, G/B>{self.easy_params['ratGb']}",
                font=("Arial", 10)
            )
            self.easy_params_label.pack(pady=2)
            
            params_button_row = ctk.CTkFrame(easy_params_frame)
            params_button_row.pack(pady=2)
            
            self._make_ctk_button(
                params_button_row, 
                "adjust_easy_params",
                text="파라미터 조정", 
                command=self.adjust_easy_params,
                fg_color=("gray", "gray30"),
                width=120
            ).pack(side="left", padx=2)
            
            self._make_ctk_button(
                params_button_row, 
                "reset_auto_params",
                text="자동 리셋", 
                command=self.reset_auto_params,
                fg_color=("orange", "darkorange"),
                width=80
            ).pack(side="left", padx=2)
            
            # 배경색 및 Scale 색상 선택 옵션
            color_options_frame = ctk.CTkFrame(easy_params_frame)
            color_options_frame.pack(pady=5, fill="x")
            
            ctk.CTkLabel(color_options_frame, text="배경색:", width=50).pack(side="left", padx=2)
            self.background_color_var = tk.StringVar(value="dark")
            self.background_color_menu = ctk.CTkOptionMenu(
                color_options_frame,
                variable=self.background_color_var,
                values=["dark", "white"],
                width=70
            )
            self.background_color_menu.pack(side="left", padx=2)
            
            ctk.CTkLabel(color_options_frame, text="Scale:", width=45).pack(side="left", padx=(5, 2))
            self.scale_color_var = tk.StringVar(value="red")
            self.scale_color_menu = ctk.CTkOptionMenu(
                color_options_frame,
                variable=self.scale_color_var,
                values=["red", "blue"],
                width=70
            )
            self.scale_color_menu.pack(side="left", padx=2)
            
            # 기본 분석 버튼
            self.basic_analyze_button = self._make_ctk_button(
                easy_params_frame, 
                "basic_analyze",
                text="기본 분석 (빠른 색상 기반)", 
                command=self.basic_analyze,
                fg_color=("green", "darkgreen")
            )
            self.basic_analyze_button.pack(pady=5, fill="x")
            
            # 고급 분석
            self.analyze_button = self._make_ctk_button(
                analysis_frame, 
                "analyze_image",
                text="고급 분석", 
                command=self.analyze_image,
                fg_color=("blue", "darkblue")
            )
            self.analyze_button.pack(pady=5, fill="x")

            # SAM3 혼합 분석
            sam3_frame = ctk.CTkFrame(analysis_frame)
            sam3_frame.pack(fill="x", pady=(4, 6))
            ctk.CTkLabel(sam3_frame, text="SAM3 Mixed", font=("Arial", 12, "bold")).pack(pady=(4, 2))

            self.sam3_prompt_var = tk.StringVar(value="leaf")
            prompt_row = ctk.CTkFrame(sam3_frame)
            prompt_row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(prompt_row, text="프롬프트:", width=70).pack(side="left")
            self.sam3_prompt_entry = ctk.CTkEntry(prompt_row, textvariable=self.sam3_prompt_var)
            self.sam3_prompt_entry.pack(side="left", fill="x", expand=True, padx=2)

            self.sam3_score_threshold_var = tk.DoubleVar(value=0.4)
            score_row = ctk.CTkFrame(sam3_frame)
            score_row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(score_row, text="점수 임계값:", width=90).pack(side="left")
            self.sam3_score_entry = ctk.CTkEntry(score_row, textvariable=self.sam3_score_threshold_var, width=80)
            self.sam3_score_entry.pack(side="left", padx=2)
            ctk.CTkLabel(score_row, text="(0~1)", text_color="gray").pack(side="left", padx=2)

            self.sam3_analyze_button = self._make_ctk_button(
                sam3_frame,
                "mixed_analyze_sam3",
                text="혼합 분석 (SAM3)",
                command=self.mixed_analyze_sam3,
                fg_color=("purple", "#553377")
            )
            self.sam3_analyze_button.pack(pady=5, fill="x")
            
            # 최소 객체 면적(픽셀) + 재탐색
            min_area_frame = ctk.CTkFrame(sam3_frame)
            min_area_frame.pack(fill="x", padx=4, pady=(2, 6))
            ctk.CTkLabel(min_area_frame, text="최소 객체 면적(px):", width=110, anchor="w").pack(side="left")
            self.min_object_area_var = tk.StringVar(value=str(self.settings.get("min_object_area", 1000)))
            self.min_object_area_entry = ctk.CTkEntry(min_area_frame, textvariable=self.min_object_area_var, width=80)
            self.min_object_area_entry.pack(side="left", padx=4)
            self._make_ctk_button(min_area_frame, "apply_min_object_area", text="재탐색", width=70, command=self.apply_min_object_area_setting).pack(side="left")
            
            # 객체 삭제 기능 설정
            object_control_frame = ctk.CTkFrame(analysis_frame)
            object_control_frame.pack(fill="x", pady=5)
            
            # 객체 삭제 기능 체크박스
            self.object_deletion_enabled = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                object_control_frame, 
                text="객체 삭제 기능 활성화", 
                variable=self.object_deletion_enabled,
                command=self.toggle_object_deletion
            ).pack(anchor="w", pady=2)
            
            # 사용법 안내
            info_text = (
                "• Ctrl+클릭: 객체 삭제/복원\n"
                "• 우클릭: 시드 제거"
            )
            ctk.CTkLabel(
                object_control_frame, 
                text=info_text,
                font=("Arial", 9),
                justify="left",
                text_color="gray"
            ).pack(pady=2, padx=5)
            
            # 토글 가능한 수동 파라미터 패널
            self.setup_manual_parameters_toggle()
            
            # --- 분리/병합 모드 ---
            split_merge_frame = ctk.CTkFrame(self.left_frame)
            split_merge_frame.pack(fill="x", pady=8)
            ctk.CTkLabel(split_merge_frame, text="객체 분리/병합", font=("Arial", 14, "bold")).pack(pady=(6,2))
            mode_row = ctk.CTkFrame(split_merge_frame)
            mode_row.pack(fill="x", pady=2)
            self.split_mode_btn = self._make_ctk_button(
                mode_row, "toggle_split_mode", text="분리 모드 진입", command=self.toggle_split_mode, fg_color=("purple", "#553377")
            )
            self.split_mode_btn.pack(side="left", padx=2, fill="x", expand=True)
            self.merge_mode_btn = self._make_ctk_button(
                mode_row, "toggle_merge_mode", text="병합 모드 진입", command=self.toggle_merge_mode, fg_color=("#0a7", "#095")
            )
            self.merge_mode_btn.pack(side="left", padx=2, fill="x", expand=True)
            action_row = ctk.CTkFrame(split_merge_frame)
            action_row.pack(fill="x", pady=2)
            self.split_undo_btn = self._make_ctk_button(action_row, "split_undo", text="되돌리기", width=80, command=self.split_undo)
            self.split_undo_btn.pack(side="left", padx=2)
            self.split_apply_btn = self._make_ctk_button(action_row, "split_apply", text="완료", width=80, command=self.split_apply)
            self.split_apply_btn.pack(side="left", padx=2)
            ctk.CTkLabel(
                split_merge_frame,
                text="안내: 분리=객체 클릭→시드 2점→완료 / 병합=두 객체 이상 선택→완료",
                font=("Arial", 10),
                text_color=("gray30","gray70")
            ).pack(pady=(2,6))
            
            # --- 삭제 모드 ---
            delete_frame = ctk.CTkFrame(self.left_frame)
            delete_frame.pack(fill="x", pady=8)
            ctk.CTkLabel(delete_frame, text="객체 삭제", font=("Arial", 14, "bold")).pack(pady=(6,2))
            self.delete_mode_btn = self._make_ctk_button(delete_frame, "toggle_delete_mode", text="삭제 모드 진입", command=self.toggle_delete_mode, fg_color=("#c33", "#922"))
            self.delete_mode_btn.pack(pady=2, fill="x")
            delete_btns_row = ctk.CTkFrame(delete_frame)
            delete_btns_row.pack(fill="x", pady=2)
            self.delete_apply_btn = self._make_ctk_button(delete_btns_row, "delete_apply", text="선택 삭제", width=80, command=self.delete_apply)
            self.delete_apply_btn.pack(side="left", padx=2)
            self.delete_clear_btn = self._make_ctk_button(delete_btns_row, "delete_clear", text="선택 해제", width=80, command=self.delete_clear)
            self.delete_clear_btn.pack(side="left", padx=2)
            ctk.CTkLabel(delete_frame, text="안내: 객체 클릭→여러 개 선택→'선택 삭제'", font=("Arial", 10), text_color=("gray30","gray70")).pack(pady=(2,6))
            
            # 결과 내보내기 (삭제 모드 아래로 이동)
            export_frame = ctk.CTkFrame(self.left_frame)
            export_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(export_frame, text="결과 내보내기", font=("Arial", 14, "bold")).pack(pady=5)
            # Scale 면적(cm^2) 설정
            scale_area_row = ctk.CTkFrame(export_frame)
            scale_area_row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(scale_area_row, text="Scale 면적(cm²):", width=110, anchor="w").pack(side="left")
            self.scale_area_var = tk.StringVar(value=str(self.settings.get("scale_area_cm2", 4.0)))
            self.scale_area_entry = ctk.CTkEntry(scale_area_row, textvariable=self.scale_area_var, width=80)
            self.scale_area_entry.pack(side="left", padx=4)
            self._make_ctk_button(scale_area_row, "apply_scale_area_setting", text="적용", width=60, command=self.apply_scale_area_setting).pack(side="left")
            self._make_ctk_button(export_frame, "export_csv", text="CSV로 내보내기", command=self.export_csv).pack(pady=2, fill="x")
            self._make_ctk_button(export_frame, "export_json", text="JSON으로 내보내기", command=self.export_json).pack(pady=2, fill="x")
            # YOLO 내보내기 (OBB/Seg)
            self._make_ctk_button(export_frame, "export_yolo_obb", text="YOLO OBB 내보내기", command=self.export_yolo_obb, fg_color=("#444", "#333")).pack(pady=2, fill="x")
            self._make_ctk_button(export_frame, "export_yolo_seg", text="YOLO Seg 내보내기", command=self.export_yolo_seg, fg_color=("#444", "#333")).pack(pady=2, fill="x")
            self._make_ctk_button(export_frame, "export_coco_seg", text="COCO Seg 내보내기", command=self.export_coco_seg, fg_color=("#444", "#333")).pack(pady=2, fill="x")
            
            # 객체 복원 버튼
            self._make_ctk_button(
                export_frame, 
                "reset_object_deletions",
                text="삭제된 객체 모두 복원", 
                command=self.reset_object_deletions,
                fg_color=("orange", "darkorange")
            ).pack(pady=2, fill="x")
            
            # COCO Polygon 형식 옵션 (CVAT/Roboflow 호환)
            self.use_polygon_format = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(
                export_frame, 
                text="Polygon 형식 사용 (CVAT/Roboflow 호환)", 
                variable=self.use_polygon_format
            ).pack(anchor="w", padx=20, pady=(0, 5))
            
        else:
            # tkinter 버전 (간소화됨)
            ttk.Label(self.left_frame, text="Advanced Leaf Analyzer").pack(pady=10)
            def _make_ttk_btn(parent, action_id, **kwargs):
                cmd = kwargs.pop("command", None)
                if cmd is not None:
                    cmd = self._wrap_button_command(action_id, cmd)
                btn = ttk.Button(parent, command=cmd, **kwargs)
                try:
                    self._widget_action_map[btn] = action_id
                    self._action_buttons[action_id] = btn
                except Exception:
                    pass
                return btn
            _make_ttk_btn(self.left_frame, "load_image", text="이미지 열기", command=self.load_image).pack(pady=5, fill="x")
            _make_ttk_btn(self.left_frame, "basic_analyze", text="기본 분석", command=self.basic_analyze).pack(pady=5, fill="x")
            _make_ttk_btn(self.left_frame, "analyze_image", text="고급 분석", command=self.analyze_image).pack(pady=5, fill="x")
            self.sam3_prompt_var = tk.StringVar(value="leaf")
            self.sam3_score_threshold_var = tk.DoubleVar(value=0.4)
            _make_ttk_btn(self.left_frame, "mixed_analyze_sam3", text="혼합 분석 (SAM3)", command=self.mixed_analyze_sam3).pack(pady=5, fill="x")
            resize_row = ttk.Frame(self.left_frame)
            resize_row.pack(fill="x", pady=4)
            ttk.Label(resize_row, text="추론 리사이즈 배율:").pack(side="left")
            self.inference_resize_var = tk.StringVar(value=str(self.settings.get("inference_resize_divisor", 1)))
            self.inference_resize_entry = ttk.Entry(resize_row, textvariable=self.inference_resize_var, width=8)
            self.inference_resize_entry.pack(side="left", padx=4)
            _make_ttk_btn(resize_row, "apply_inference_resize", text="적용", command=self.apply_inference_resize_setting).pack(side="left")
            # 최소 객체 면적(픽셀) + 재탐색
            min_area_row = ttk.Frame(self.left_frame)
            min_area_row.pack(fill="x", pady=4)
            ttk.Label(min_area_row, text="최소 객체 면적(px):").pack(side="left")
            self.min_object_area_var = tk.StringVar(value=str(self.settings.get("min_object_area", 1000)))
            self.min_object_area_entry = ttk.Entry(min_area_row, textvariable=self.min_object_area_var, width=8)
            self.min_object_area_entry.pack(side="left", padx=4)
            _make_ttk_btn(min_area_row, "apply_min_object_area", text="재탐색", command=self.apply_min_object_area_setting).pack(side="left")

    def setup_canvas(self):
        """이미지 표시 캔버스 설정"""
        if CTK_AVAILABLE:
            self.canvas_frame = ctk.CTkFrame(self.right_frame)
        else:
            self.canvas_frame = ttk.Frame(self.right_frame)
            
        self.canvas_frame.pack(fill="both", expand=True)
        
        # 캔버스
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill="both", expand=True)
        
        # 캔버스 이벤트 바인딩 (최소 로그)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        # 객체 삭제용 이벤트 (Ctrl+클릭만 사용)
        self.canvas.bind("<Control-Button-1>", self.on_object_delete_click)
        # 분리 모드 안내선(시드 2점) 미리보기는 최소 구현: 점 2개는 팝업으로 안내

    def setup_manual_parameters_toggle(self):
        """토글 가능한 수동 파라미터 패널 설정"""
        if not CTK_AVAILABLE:
            return  # CustomTkinter에서만 지원
            
        # 토글 버튼
        self.manual_toggle_btn = self._make_ctk_button(
            self.left_frame,
            "toggle_manual_parameters",
            text="▶ 수동 파라미터 조정",
            command=self.toggle_manual_parameters,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            hover_color=("gray80", "gray30"),
            anchor="w"
        )
        self.manual_toggle_btn.pack(fill="x", pady=(10, 0))
        
        # 수동 파라미터 프레임 생성 (처음에는 숨김)
        self.manual_params_frame = ctk.CTkFrame(self.left_frame)
        self.manual_params_frame.pack_forget()  # 명시적으로 숨김
        
        # 컨트롤들 미리 생성
        self.setup_manual_parameter_controls()

    def setup_manual_parameter_controls(self):
        """수동 파라미터 컨트롤들 설정 (GrabCut 이후 후처리만 유지)"""
        if not CTK_AVAILABLE or not self.manual_params_frame:
            return
        title_label = ctk.CTkLabel(
            self.manual_params_frame, 
            text="후처리 파라미터",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(10, 5))
        
        control_frame = ctk.CTkFrame(self.manual_params_frame)
        control_frame.pack(fill="x", pady=5, padx=5)
        self.manual_preview_var = tk.BooleanVar(value=self.manual_settings["manual_preview"])
        self.manual_preview_var.trace_add("write", lambda n, i, m: self.on_preview_toggle())
        ctk.CTkCheckBox(
            control_frame, 
            text="실시간 미리보기", 
            variable=self.manual_preview_var,
            command=self.on_preview_toggle
        ).pack(anchor="w", padx=5, pady=2)
        
        manual_analyze_btn = self._make_ctk_button(
            control_frame,
            "manual_analyze",
            text="수동 분석 실행",
            command=self.manual_analyze,
            fg_color=("blue", "darkblue")
        )
        manual_analyze_btn.pack(fill="x", padx=5, pady=5)

    def toggle_manual_parameters(self):
        """수동 파라미터 패널 토글"""
        if not CTK_AVAILABLE:
            return
            
        print(f"토글 클릭됨! 현재 상태: {self.manual_params_visible}")  # 디버그
        
        self.manual_params_visible = not self.manual_params_visible
        
        if self.manual_params_visible:
            # 패널 표시
            self.manual_params_frame.pack(fill="x", pady=5, after=self.manual_toggle_btn)
            self.manual_toggle_btn.configure(text="▼ 수동 파라미터 조정")
            print("수동 파라미터 패널을 표시했습니다.")  # 디버그
        else:
            # 패널 숨김
            self.manual_params_frame.pack_forget()
            self.manual_toggle_btn.configure(text="▶ 수동 파라미터 조정")
            print("수동 파라미터 패널을 숨겼습니다.")  # 디버그
            
        # 업데이트 강제 적용
        self.left_frame.update()
        print(f"토글 완료. 새 상태: {self.manual_params_visible}")  # 디버그

    def on_parameter_change(self, value):
        """파라미터 슬라이더 변경 시 호출"""
        if not CTK_AVAILABLE:
            return
        # 값 업데이트 (존재하는 컨트롤만 반영)
        if hasattr(self, "morph_kernel_var"):
            self.manual_settings["morph_kernel"] = int(self.morph_kernel_var.get())
            self.settings["morphology_kernel_size"] = int(self.morph_kernel_var.get())
            if hasattr(self, "morph_kernel_value_label"):
                self.morph_kernel_value_label.configure(text=f"{self.manual_settings['morph_kernel']}")
        
        # 실시간 미리보기가 활성화되어 있으면 자동 업데이트
        preview_enabled = (hasattr(self, 'manual_preview_var') and self.manual_preview_var.get())
        image_loaded = self.original_image is not None
        
        print(f"실시간 미리보기 활성화: {preview_enabled}")  # 디버그
        print(f"이미지 로드됨: {image_loaded}")  # 디버그
        
        if preview_enabled and image_loaded:
            print("실시간 미리보기 업데이트 시작...")  # 디버그
            # 필터만 갱신: GrabCut 재실행 금지, Canvas 크기 유지
            self.update_manual_preview(filter_only=True)
        else:
            print("실시간 미리보기 건너뜀")  # 디버그

    def on_preview_toggle(self):
        """실시간 미리보기 토글"""
        preview_state = self.manual_preview_var.get()
        self.manual_settings["manual_preview"] = preview_state
        
        print(f"실시간 미리보기 토글: {preview_state}")  # 디버그
        
        if preview_state and self.original_image is not None:
            print("미리보기 활성화 - 업데이트 실행")  # 디버그
            self.update_manual_preview()
        else:
            print(f"미리보기 비활성화 또는 이미지 없음 (이미지: {self.original_image is not None})")  # 디버그

    def update_manual_preview(self, filter_only: bool = False):
        """수동 파라미터로 미리보기 업데이트 (캐시 활용)
        filter_only=True일 때 GrabCut/세그멘테이션 재실행 없이 캐시/기존 결과에만 면적 필터 재적용
        """
        print("update_manual_preview 함수 시작")  # 디버그
        
        # 이미지 데이터 확인
        if self.original_image is None:
            return
            
        try:
            # 시드 상태 확인
            leaf_seeds = self.seed_manager.seeds.get("leaf", [])
            
            # 필터 전용 모드: GrabCut 등 재실행 금지, 기존 결과/캐시만 사용
            if filter_only:
                # Leaf 마스크 소스 결정 (최종 분석 결과 우선, 그 다음 캐시)
                base_mask = None
                if (hasattr(self, 'analysis_results') and self.analysis_results and 'leaf_mask' in self.analysis_results):
                    base_mask = self.analysis_results['leaf_mask']
                elif self._cached_raw_mask is not None:
                    base_mask = self._cached_raw_mask
                
                if base_mask is None:
                    print("filter_only 모드: 사용할 Leaf 마스크가 없어 스킵")
                    return
                
                # Leaf 삭제 상태 반영: 인스턴스 라벨맵을 사용해 삭제된 ID 제외
                if self._current_instance_labels is not None and len(self._deleted_objects) > 0:
                    keep_mask = np.ones_like(self._current_instance_labels, dtype=bool)
                    for oid in self._deleted_objects:
                        keep_mask[self._current_instance_labels == oid] = False
                    if keep_mask.shape != base_mask.shape:
                        try:
                            keep_mask = cv2.resize(keep_mask.astype(np.uint8), (base_mask.shape[1], base_mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        except Exception:
                            pass
                    base_mask = base_mask & keep_mask
                
                filtered_mask = self._apply_size_filter(base_mask)
                
                # Scale 마스크 소스 결정 + 삭제 반영 + 면적 필터
                filtered_scale_mask = None
                if self._current_scale_labels is not None:
                    filtered_scale_mask = self._create_filtered_scale_mask()
                    if filtered_scale_mask.size > 0:
                        filtered_scale_mask = self._apply_size_filter(filtered_scale_mask)
                else:
                    src_scale = None
                    if self._cached_scale_mask is not None:
                        src_scale = self._cached_scale_mask
                    elif (hasattr(self, 'analysis_results') and self.analysis_results and 'scale_mask' in self.analysis_results):
                        src_scale = self.analysis_results['scale_mask']
                    if src_scale is not None and np.sum(src_scale) > 0:
                        filtered_scale_mask = self._apply_size_filter(src_scale)
                
                # 미리보기 생성/표시
                print("filter_only 모드: 오버레이 생성 시작...")
                stats = self._build_stats_from_masks(filtered_mask, filtered_scale_mask, leaf_area_thresh=1, scale_area_thresh=1)
                preview_image = self.create_preview_overlay(filtered_mask, filtered_scale_mask, stats)
                if preview_image is not None:
                    self.show_preview_image(preview_image)
                    print("filter_only 모드: 미리보기 업데이트 완료!")
                return
            
            # 시드가 없으면 기본 분석 결과 재사용 (삭제 반영 + 면적 필터만 적용)
            if len(leaf_seeds) == 0:
                print("시드가 없어 기본 분석 결과 재사용")
                
                # 기본 분석 결과가 있으면 그것을 사용
                if (hasattr(self, 'analysis_results') and 
                    self.analysis_results and 
                    'leaf_mask' in self.analysis_results):
                    
                    # 기존 결과에 현재 파라미터 적용 (Leaf)
                    base_mask = self.analysis_results['leaf_mask']
                    filtered_mask = self._apply_size_filter(base_mask)
                    
                    # Scale 마스크: 삭제 반영된 마스크 우선
                    if self._current_scale_labels is not None:
                        scale_mask = self._create_filtered_scale_mask()
                        if scale_mask.size > 0:
                            scale_mask = self._apply_size_filter(scale_mask)
                    else:
                        scale_mask = self.analysis_results.get('scale_mask')
                        if scale_mask is not None and np.sum(scale_mask) > 0:
                            scale_mask = self._apply_size_filter(scale_mask)
                    
                    # 미리보기 생성
                    stats = self._build_stats_from_masks(filtered_mask, scale_mask, leaf_area_thresh=1, scale_area_thresh=1)
                    preview_image = self.create_preview_overlay(filtered_mask, scale_mask, stats)
                    if preview_image is not None:
                        self.show_preview_image(preview_image)
                        print("기본 분석 결과 기반 미리보기 업데이트 완룼")
                else:
                    print("기본 분석 결과가 없어 미리보기 스킵")
                return
            
            # 시드가 있으면 GrabCut 기반 미리보기
            current_seed_signature = self._get_seed_signature()
            seeds_changed = (current_seed_signature != self._last_seed_signature)
            
            # 시드가 변경되었거나 캐시가 없으면 마스크 재생성
            if seeds_changed or self._cached_raw_mask is None:
                print("시드 변경 감지 - 마스크 재생성...")
                self._cached_raw_mask = self.generate_leaf_mask()
                
                # Scale 마스크는 시드가 있을 때만 생성 (모든 모드에서 통일된 로직)
                scale_seeds = self.seed_manager.seeds.get("scale", [])
                if len(scale_seeds) > 0:
                    print(f"   → 미리보기: Scale seed {len(scale_seeds)}개 기반 마스크 생성")
                    if self._get_segmentation_method() == "superpixel" and self.superpixel_labels is not None:
                        self._cached_scale_mask = self._generate_scale_mask_superpixel()
                    else:
                        self._cached_scale_mask = self._generate_scale_mask_from_seeds(scale_seeds)
                else:
                    print("   → 미리보기: Scale seed 없음 - Scale 마스크 스킵")
                    self._cached_scale_mask = None
                    
                self._last_seed_signature = current_seed_signature
                print(f"새 마스크 생성: {np.sum(self._cached_raw_mask)} 픽셀")
            else:
                print("캐시된 마스크 사용 - 세그멘테이션 스킵")
            
            # 캐시된 마스크에 현재 파라미터 적용
            filtered_mask = self._apply_size_filter(self._cached_raw_mask)
            
            # Scale 마스크에도 동일한 면적 필터링 적용
            filtered_scale_mask = None
            if self._cached_scale_mask is not None and np.sum(self._cached_scale_mask) > 0:
                filtered_scale_mask = self._apply_size_filter(self._cached_scale_mask)
                print(f"   → Scale 마스크도 면적 필터링: {np.sum(self._cached_scale_mask)} → {np.sum(filtered_scale_mask)}픽셀")
            
            print("미리보기 오버레이 생성 시작...")
            # 미리보기 이미지 생성
            stats = self._build_stats_from_masks(filtered_mask, filtered_scale_mask, leaf_area_thresh=1, scale_area_thresh=1)
            preview_image = self.create_preview_overlay(filtered_mask, filtered_scale_mask, stats)
            
            if preview_image is None:
                print("오버레이 생성 실패!")
                return
            print(f"미리보기 이미지 크기: {preview_image.shape}")
            
            print("캔버스에 표시 시작...")
            # 캔버스에 표시
            self.show_preview_image(preview_image)
            print("미리보기 업데이트 완료!")
            
        except Exception as e:
            print(f"미리보기 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()


    def update_display_image(self):
        """표시 이미지 업데이트"""
        # 표시할 이미지 선택 (보정된 이미지가 있으면 우선, 없으면 원본)
        working_image = getattr(self, 'working_image', None)
        original_image = getattr(self, 'original_image', None)
        
        # 안전한 None 체크 (numpy 배열 호환)
        if working_image is not None and hasattr(working_image, 'shape'):
            source_image = working_image
        elif original_image is not None and hasattr(original_image, 'shape'):
            source_image = original_image
        else:
            return
        
        # 이미지 크기 조정 (캔버스에 맞춤)
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
        except:
            return
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.update_display_image)
            return
        
        h, w = source_image.shape[:2]
        
        # 비율 유지하며 크기 조정
        scale = min(canvas_width/w, canvas_height/h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        self.display_image = cv2.resize(source_image, (new_w, new_h))
        self.display_scale = scale
        
        # 시드 마커 추가
        display_with_seeds = self.add_seed_markers(self.display_image.copy())
        # 분리 모드 강조: dim 처리
        if getattr(self, 'split_mode_enabled', False):
            display_with_seeds = (display_with_seeds * 0.92).astype(np.uint8)
        
        # PIL 이미지로 변환하여 표시
        pil_image = Image.fromarray(display_with_seeds)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 캔버스에 표시
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
