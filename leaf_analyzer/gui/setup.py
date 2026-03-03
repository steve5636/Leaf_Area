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
            self.root.title("Leaf Area Analyzer")
            self.root.geometry("1400x900")
        else:
            self.root = tk.Tk()
            self.root.title("Leaf Area Analyzer")
            self.root.geometry("1400x900")
        self._warn_if_ctk_missing()
        
        # Return/Enter 키 중복 입력 방지
        # (messagebox 닫기 직후 Enter 재입력으로 버튼이 연속 실행되는 문제 완화)
        self._block_return_key = False
        def _on_return(event):
            if self._block_return_key:
                return "break"  # 이벤트 전파 중단
            return None
        self.root.bind("<Return>", _on_return)
        self.root.bind("<KP_Enter>", _on_return)  # 숫자패드 Enter

        # 버튼은 각 위젯의 native command 경로로 실행한다.
        # (전역 ButtonPress 훅 기반 라우팅은 macOS 비활성 창 상태에서
        # 잘못된 위젯 이벤트를 유발해 이전 동작이 재실행될 수 있음)
        def _wrap_button_command(_action_id, func):
            return func

        self._wrap_button_command = _wrap_button_command

        def _make_ctk_button(parent, action_id, **kwargs):
            _ = action_id
            cmd = kwargs.pop("command", None)
            btn = ctk.CTkButton(parent, command=cmd, **kwargs)
            return btn
        self._make_ctk_button = _make_ctk_button

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

            # 배치 처리 옵션
            batch_opts = ctk.CTkFrame(file_frame)
            batch_opts.pack(fill="x", padx=4, pady=(2, 4))
            ctk.CTkLabel(batch_opts, text="배치 분석 모드:", width=90, anchor="w").pack(side="left")
            self.batch_analysis_mode_var = tk.StringVar(value="sam3")
            self.batch_analysis_mode_menu = ctk.CTkOptionMenu(
                batch_opts,
                variable=self.batch_analysis_mode_var,
                values=["sam3", "basic"],
                width=110
            )
            self.batch_analysis_mode_menu.pack(side="left", padx=2)
            self.batch_save_per_image_var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                file_frame,
                text="배치: 이미지별 CSV/JSON 저장",
                variable=self.batch_save_per_image_var
            ).pack(anchor="w", padx=8, pady=(0, 4))
            self.batch_export_yolo_coco_var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                file_frame,
                text="배치: YOLO/COCO 같이 내보내기",
                variable=self.batch_export_yolo_coco_var
            ).pack(anchor="w", padx=8, pady=(0, 6))

            # 배치 리뷰 네비게이션
            self.batch_review_label = ctk.CTkLabel(
                file_frame,
                text="배치 리뷰: 없음",
                font=("Arial", 10),
                text_color="gray"
            )
            self.batch_review_label.pack(anchor="w", padx=8, pady=(0, 2))
            review_row = ctk.CTkFrame(file_frame)
            review_row.pack(fill="x", padx=4, pady=(0, 4))
            self.batch_review_prev_btn = self._make_ctk_button(
                review_row, "batch_review_prev", text="이전", width=60, command=self.batch_review_prev
            )
            self.batch_review_prev_btn.pack(side="left", padx=2)
            self.batch_review_save_btn = self._make_ctk_button(
                review_row, "batch_review_save_current", text="누적 저장", width=90, command=self.batch_review_save_current
            )
            self.batch_review_save_btn.pack(side="left", padx=2)
            self.batch_review_next_btn = self._make_ctk_button(
                review_row, "batch_review_next", text="다음", width=60, command=self.batch_review_next
            )
            self.batch_review_next_btn.pack(side="left", padx=2)
            try:
                self.batch_review_prev_btn.configure(state="disabled")
                self.batch_review_save_btn.configure(state="disabled")
                self.batch_review_next_btn.configure(state="disabled")
            except Exception:
                pass

            # Scale 면적 환산 설정 (내보내기 핵심값: 상단 고정)
            scale_area_frame = ctk.CTkFrame(self.left_frame)
            scale_area_frame.pack(fill="x", pady=5)
            ctk.CTkLabel(scale_area_frame, text="면적 환산 설정", font=("Arial", 14, "bold")).pack(pady=5)
            scale_area_row = ctk.CTkFrame(scale_area_frame)
            scale_area_row.pack(fill="x", padx=4, pady=(2, 4))
            ctk.CTkLabel(scale_area_row, text="Scale 면적(cm²):", width=110, anchor="w").pack(side="left")
            self.scale_area_var = tk.StringVar(value=str(self.settings.get("scale_area_cm2", 4.0)))
            self.scale_area_entry = ctk.CTkEntry(scale_area_row, textvariable=self.scale_area_var, width=80)
            self.scale_area_entry.pack(side="left", padx=4)
            self._make_ctk_button(
                scale_area_row,
                "apply_scale_area_setting",
                text="적용",
                width=60,
                command=self.apply_scale_area_setting,
            ).pack(side="left")
            ctk.CTkLabel(
                scale_area_frame,
                text="CSV/JSON/YOLO/COCO 내보내기 cm² 환산에 사용",
                font=("Arial", 10),
                text_color=("gray35", "gray70"),
            ).pack(anchor="w", padx=8, pady=(0, 4))
            
            # SAM3 후보정 시드
            seed_frame = ctk.CTkFrame(self.left_frame)
            seed_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(seed_frame, text="SAM3 누락 객체 추가", font=("Arial", 14, "bold")).pack(pady=5)

            self.seed_edit_enabled_var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                seed_frame,
                text="포인트 입력 모드",
                variable=self.seed_edit_enabled_var
            ).pack(anchor="w")

            self.seed_mode = ctk.StringVar(value="leaf")
            ctk.CTkRadioButton(seed_frame, text="Leaf (+)", variable=self.seed_mode, value="leaf").pack(anchor="w")
            ctk.CTkRadioButton(seed_frame, text="Plant (+)", variable=self.seed_mode, value="plant").pack(anchor="w")
            ctk.CTkRadioButton(seed_frame, text="Scale (+)", variable=self.seed_mode, value="scale").pack(anchor="w")
            ctk.CTkRadioButton(seed_frame, text="배경 (-)", variable=self.seed_mode, value="background").pack(anchor="w")
            
            self._make_ctk_button(seed_frame, "clear_current_seeds", text="시드 초기화", command=self.clear_current_seeds).pack(pady=2, fill="x")
            self._make_ctk_button(seed_frame, "undo_last_seed", text="실행 취소 (Undo)", command=self.undo_last_seed).pack(pady=2, fill="x")
            self._make_ctk_button(
                seed_frame,
                "apply_sam3_seed_correction",
                text="누락 객체 추가 (SAM3)",
                command=self.apply_sam3_seed_correction,
                fg_color=("blue", "#2d588f")
            ).pack(pady=(4, 2), fill="x")
            ctk.CTkLabel(
                seed_frame,
                text="Leaf/Plant/Scale 중 한 종류와 배경(-) 시드를 함께 찍어 보정",
                font=("Arial", 10),
                text_color=("gray35", "gray70")
            ).pack(anchor="w", pady=(0, 4))
            
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
            
            # 결과 내보내기
            export_frame = ctk.CTkFrame(self.left_frame)
            export_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(export_frame, text="결과 내보내기", font=("Arial", 14, "bold")).pack(pady=5)
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
            self.csv_include_pixels_var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                export_frame,
                text="CSV에 픽셀값 포함",
                variable=self.csv_include_pixels_var
            ).pack(anchor="w", padx=20, pady=(0, 5))
            
        else:
            # tkinter 버전 (간소화됨)
            ttk.Label(self.left_frame, text="Leaf Area Analyzer").pack(pady=10)
            def _make_ttk_btn(parent, action_id, **kwargs):
                _ = action_id
                cmd = kwargs.pop("command", None)
                btn = ttk.Button(parent, command=cmd, **kwargs)
                return btn
            _make_ttk_btn(self.left_frame, "load_image", text="이미지 열기", command=self.load_image).pack(pady=5, fill="x")
            _make_ttk_btn(self.left_frame, "batch_process", text="배치 처리", command=self.batch_process).pack(pady=5, fill="x")
            scale_area_row = ttk.Frame(self.left_frame)
            scale_area_row.pack(fill="x", pady=4)
            ttk.Label(scale_area_row, text="Scale 면적(cm²):").pack(side="left")
            self.scale_area_var = tk.StringVar(value=str(self.settings.get("scale_area_cm2", 4.0)))
            self.scale_area_entry = ttk.Entry(scale_area_row, textvariable=self.scale_area_var, width=8)
            self.scale_area_entry.pack(side="left", padx=4)
            _make_ttk_btn(scale_area_row, "apply_scale_area_setting", text="적용", command=self.apply_scale_area_setting).pack(side="left")
            _make_ttk_btn(self.left_frame, "basic_analyze", text="기본 분석", command=self.basic_analyze).pack(pady=5, fill="x")
            self.batch_analysis_mode_var = tk.StringVar(value="sam3")
            ttk.Label(self.left_frame, text="배치 분석 모드").pack(pady=(6, 0))
            ttk.OptionMenu(self.left_frame, self.batch_analysis_mode_var, "sam3", "sam3", "basic").pack(pady=2, fill="x")
            self.batch_save_per_image_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.left_frame, text="배치: 이미지별 CSV/JSON 저장", variable=self.batch_save_per_image_var).pack(pady=(0, 6), anchor="w")
            self.batch_export_yolo_coco_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.left_frame, text="배치: YOLO/COCO 같이 내보내기", variable=self.batch_export_yolo_coco_var).pack(pady=(0, 6), anchor="w")
            self.csv_include_pixels_var = tk.BooleanVar(value=False)
            ttk.Label(self.left_frame, text="SAM3 후보정 포인트").pack(pady=(2, 0), anchor="w")
            self.seed_edit_enabled_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.left_frame, text="포인트 입력 모드", variable=self.seed_edit_enabled_var).pack(pady=(0, 2), anchor="w")
            self.seed_mode = tk.StringVar(value="leaf")
            ttk.Radiobutton(self.left_frame, text="Leaf (+)", variable=self.seed_mode, value="leaf").pack(anchor="w")
            ttk.Radiobutton(self.left_frame, text="Plant (+)", variable=self.seed_mode, value="plant").pack(anchor="w")
            ttk.Radiobutton(self.left_frame, text="Scale (+)", variable=self.seed_mode, value="scale").pack(anchor="w")
            ttk.Radiobutton(self.left_frame, text="배경 (-)", variable=self.seed_mode, value="background").pack(anchor="w")
            _make_ttk_btn(self.left_frame, "clear_current_seeds", text="시드 초기화", command=self.clear_current_seeds).pack(pady=2, fill="x")
            _make_ttk_btn(self.left_frame, "undo_last_seed", text="실행 취소 (Undo)", command=self.undo_last_seed).pack(pady=2, fill="x")
            _make_ttk_btn(self.left_frame, "apply_sam3_seed_correction", text="누락 객체 추가 (SAM3)", command=self.apply_sam3_seed_correction).pack(pady=2, fill="x")
            self.batch_review_label = ttk.Label(self.left_frame, text="배치 리뷰: 없음")
            self.batch_review_label.pack(pady=(0, 2), anchor="w")
            review_row = ttk.Frame(self.left_frame)
            review_row.pack(fill="x", pady=(0, 6))
            self.batch_review_prev_btn = _make_ttk_btn(review_row, "batch_review_prev", text="이전", command=self.batch_review_prev)
            self.batch_review_prev_btn.pack(side="left", padx=2)
            self.batch_review_save_btn = _make_ttk_btn(review_row, "batch_review_save_current", text="누적 저장", command=self.batch_review_save_current)
            self.batch_review_save_btn.pack(side="left", padx=2)
            self.batch_review_next_btn = _make_ttk_btn(review_row, "batch_review_next", text="다음", command=self.batch_review_next)
            self.batch_review_next_btn.pack(side="left", padx=2)
            try:
                self.batch_review_prev_btn.configure(state="disabled")
                self.batch_review_save_btn.configure(state="disabled")
                self.batch_review_next_btn.configure(state="disabled")
            except Exception:
                pass
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
