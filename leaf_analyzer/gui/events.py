#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Handlers for Advanced Leaf Analyzer
이벤트 핸들러
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from scipy import ndimage
from PIL import Image, ImageTk
from skimage.segmentation import watershed
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False


class EventHandlers:
    """이벤트 핸들러 믹스인 클래스"""

    def _event_to_orig_coords(self, event, log: bool = False):
        """캔버스 이벤트 좌표를 원본 이미지 좌표로 변환. 실패 시 None 반환."""
        try:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            if log:
                self._log(f" 캔버스 좌표: ({canvas_x:.1f}, {canvas_y:.1f})")
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if log:
                self._log(f" 캔버스 크기: {canvas_width} × {canvas_height}")
        except Exception:
            return None
        
        if not hasattr(self, 'display_image') or self.display_image is None:
            if log:
                self._log(" display_image가 없음")
            return None
        
        img_w, img_h = self.display_image.shape[1], self.display_image.shape[0]
        if log:
            self._log(f" 이미지 크기: {img_w} × {img_h}")
        
        if not hasattr(self, 'display_scale') or self.display_scale is None or self.display_scale == 0:
            if log:
                self._log(" display_scale이 없음")
            return None
        if log:
            self._log(f"🔍 display_scale: {self.display_scale}")
        
        left = (canvas_width - img_w) // 2
        top = (canvas_height - img_h) // 2
        if (canvas_x < left or canvas_x > left + img_w or 
            canvas_y < top or canvas_y > top + img_h):
            if log:
                self._log(f" 이미지 영역 외부 클릭 (이미지 영역: {left}, {top}, {left + img_w}, {top + img_h})")
            return None
        
        rel_x = canvas_x - left
        rel_y = canvas_y - top
        orig_x = int(rel_x / self.display_scale)
        orig_y = int(rel_y / self.display_scale)
        return (orig_x, orig_y)

    def on_canvas_click(self, event):
        """캔버스 클릭 이벤트"""
        self._log(f"캔버스 클릭 이벤트 발생: ({event.x}, {event.y})")
        
        try:
            # --- 삭제 모드 라우팅 ---
            if getattr(self, 'delete_mode_enabled', False):
                if not self.analysis_results:
                    messagebox.showwarning("경고", "먼저 분석을 실행하세요.")
                    self._safe_refocus()
                    return
                coords = self._event_to_orig_coords(event)
                if coords is None:
                    return
                orig_x, orig_y = coords
    
                hit = self._find_object_at_position(orig_x, orig_y)
                if hit is None:
                    messagebox.showinfo("정보", "객체 내부를 클릭해 선택/해제하세요.")
                    self._safe_refocus()
                    return
                obj_type, obj_id = hit
                key = (obj_type, int(obj_id))
                if key in self.delete_selected:
                    self.delete_selected.remove(key)
                else:
                    self.delete_selected.add(key)
                self.show_result_overlay()
                return
            # --- 병합 모드 라우팅 ---
            if getattr(self, 'merge_mode_enabled', False):
                if not self.analysis_results:
                    messagebox.showwarning("경고", "먼저 분석을 실행하세요.")
                    self._safe_refocus()
                    return
                coords = self._event_to_orig_coords(event)
                if coords is None:
                    return
                orig_x, orig_y = coords
    
                hit = self._find_object_at_position(orig_x, orig_y)
                if hit is None:
                    messagebox.showinfo("정보", "객체 내부를 클릭해 선택/해제하세요.")
                    self._safe_refocus()
                    return
                obj_type, obj_id = hit
                # Leaf/Scale 혼합 가드
                if len(self.merge_selected) > 0:
                    types_in_set = {t for (t, _) in self.merge_selected}
                    if obj_type not in types_in_set:
                        messagebox.showwarning("경고", "Leaf와 Scale을 함께 병합할 수 없습니다.")
                        self._safe_refocus()
                        return
                key = (obj_type, int(obj_id))
                if key in self.merge_selected:
                    self.merge_selected.remove(key)
                else:
                    self.merge_selected.add(key)
                # 선택 강조 미리보기
                self._preview_merge_result()
                return
            # --- 분리 모드 라우팅 ---
            if getattr(self, 'split_mode_enabled', False):
                if not self.analysis_results:
                    messagebox.showwarning("경고", "먼저 분석을 실행하세요.")
                    self._safe_refocus()
                    return
                coords = self._event_to_orig_coords(event)
                if coords is None:
                    return
                orig_x, orig_y = coords
    
                # 1) 아직 선택 객체가 없으면 객체 선택
                if self.split_selected_object is None:
                    hit = self._find_object_at_position(orig_x, orig_y)
                    if hit is None:
                        messagebox.showinfo("정보", "객체 내부를 클릭해 선택하세요.")
                        self._safe_refocus()
                        return
                    self.split_selected_object = hit  # (type, id)
                    self.split_mode_points = []
                    messagebox.showinfo("분리 모드", "분리 기준이 될 두 지점을 연속으로 클릭하세요.")
                    self._safe_refocus()
                    # 강조: 오버레이 불투명도 상승
                    self._show_split_overlay_highlight()
                    return
    
                # 2) 시드 기록 (두 점 수집)
                if len(self.split_mode_points) < 2:
                    self.split_mode_points.append((orig_x, orig_y))
                    if len(self.split_mode_points) == 1:
                        messagebox.showinfo("분리 모드", "두 번째 지점을 클릭하세요.")
                        self._safe_refocus()
                    else:
                        # 두 점 수집 완료 → 미리보기 실행
                        self._preview_split_result()
                    return
    
            if self.original_image is None:
                self._log("이미지가 로드되지 않음")
                return
            coords = self._event_to_orig_coords(event, log=True)
            if coords is None:
                return
            orig_x, orig_y = coords
            
            # 이미지 경계 확인
            h, w = self.original_image.shape[:2]
            if 0 <= orig_x < w and 0 <= orig_y < h:
                # seed_mode 확인
                if hasattr(self, 'seed_mode') and CTK_AVAILABLE:
                    current_mode = self.seed_mode.get()
                    self._log(f" 시드 모드: {current_mode}")
                else:
                    current_mode = "leaf"
                    self._log(f" 기본 시드 모드: {current_mode}")
                    
                self.seed_manager.current_class = current_mode
                self.seed_manager.add_seed(orig_x, orig_y)
                
                # 표시 업데이트 (시드 마커만 표시)
                self.update_display_image()
                
                # 시드 클릭 시에는 미리보기 업데이트 하지 않음 (성능 향상)
                self._log(f"{current_mode} 시드 추가됨: ({orig_x}, {orig_y})")
            else:
                self._log(f"이미지 경계 외부 좌표: ({orig_x}, {orig_y}) (이미지 크기: {w} × {h})")
                
        except Exception as e:
            self._log(f"캔버스 클릭 처리 중 오류 발생: {e}")
            try:
                import traceback
                self._log(traceback.format_exc())
            except Exception:
                pass
            self._log(" 이 오류로 인해 다른 함수가 호출되었을 수 있습니다.")

    def on_canvas_right_click(self, event):
        """우클릭으로 특정 위치의 시드 제거"""
        if self.original_image is None:
            return
        
        # 이벤트가 None이면 마지막 시드 제거 (기존 기능 유지)
        if event is None:
            self._deactivate_delete_mode()
            current_mode = self.seed_mode.get() if CTK_AVAILABLE else "leaf"
            self.seed_manager.current_class = current_mode
            self.seed_manager.remove_last_seed()
            self.update_display_image()
            return
            
        coords = self._event_to_orig_coords(event)
        if coords is None:
            return
        orig_x, orig_y = coords
        
        # 이미지 경계 확인
        h, w = self.original_image.shape[:2]
        if 0 <= orig_x < w and 0 <= orig_y < h:
            self._deactivate_delete_mode()
            current_mode = self.seed_mode.get() if CTK_AVAILABLE else "leaf"
            self.seed_manager.current_class = current_mode
            
            # 모든 클래스에서 가장 가까운 시드 찾기
            seed_removed = False
            removed_from_class = None
            
            for seed_class in ["leaf", "scale", "background"]:
                if self.seed_manager.remove_seed_at_position(orig_x, orig_y, seed_class, threshold=20):
                    seed_removed = True
                    removed_from_class = seed_class
                    break
            
            if seed_removed:
                self._log(f"{removed_from_class} 시드가 제거되었습니다.")
                
                # 마스크 캐시 무효화 (시드 변경)
                self._invalidate_mask_cache()
                
                # 객체 삭제 시스템 초기화 (시드 변경 시)
                self._deleted_objects = set()
                self._current_instance_labels = None
                
                self.update_display_image()
                
                # 시드 제거 시에도 미리보기 업데이트 하지 않음 (성능 향상)
                self._log("  → 색상 모델 재구축이 필요합니다.")
            else:
                self._log(f"클릭 위치 ({orig_x}, {orig_y}) 근처에 제거할 시드가 없습니다.")

    def on_object_delete_click(self, event):
        """객체 삭제 클릭 이벤트 (Ctrl+클릭)"""
        if not self._object_deletion_enabled or self.original_image is None:
            return
            
        if self._current_instance_labels is None:
            messagebox.showinfo("정보", "먼저 분석을 실행해주세요.")
            self._safe_refocus()
            return
            
        try:
            coords = self._event_to_orig_coords(event)
            if coords is None:
                return
            orig_x, orig_y = coords
            
            # 클릭 위치에서 객체 찾기 (Leaf 또는 Scale) - 삭제된 객체도 복원할 수 있도록 포함
            object_info = self._find_object_at_position(orig_x, orig_y, include_deleted=True)
    
            if object_info is not None:
                object_type, object_id = object_info
                
                if object_type == "leaf":
                    deleted_set = self._deleted_objects
                    type_name = "Leaf"
                else:  # "scale"
                    deleted_set = self._deleted_scale_objects
                    type_name = "Scale"
                    
                    # Scale은 1개만 유지: 선택한 Scale 외 모두 삭제 처리, 선택한 것은 항상 유지
                    if self._current_scale_labels is not None:
                        unique_scale_ids = np.unique(self._current_scale_labels)
                        # 새 삭제 집합: 선택된 ID를 제외한 모든 ID
                        self._deleted_scale_objects = set(int(sid) for sid in unique_scale_ids if sid > 0 and sid != object_id)
                        # 선택된 ID는 삭제 집합에서 제거 보장
                        if object_id in self._deleted_scale_objects:
                            self._deleted_scale_objects.discard(object_id)
                        print(f"Scale 단일 선택 모드: 선택 #{object_id} 유지, 나머지 {len(self._deleted_scale_objects)}개 삭제 표시")
                    
                    # Scale 클릭의 액션 표기는 '선택'으로 통일
                    action = "선택"
                    
                    # 메시지 표기를 위해 이후 분기 공통변수 세팅만 유지하고 토글은 수행하지 않음
                    
                if object_type != "scale":
                    if object_id in deleted_set:
                        # 이미 삭제된 객체를 다시 클릭하면 복원
                        deleted_set.remove(object_id)
                        action = "복원"
                        print(f"{type_name} 객체 {object_id} 복원")
                    else:
                        # 새로운 객체 삭제
                        deleted_set.add(object_id)
                        action = "삭제"
                        print(f"{type_name} 객체 {object_id} 삭제")
                
                # 사용자에게 피드백
                # 현재 활성 표시 개수 계산
                active_leaf = 0
                if self._current_instance_labels is not None:
                    try:
                        filtered_leaf_mask = self._create_filtered_mask()
                        active_leaf = int(np.sum(np.unique(self._current_instance_labels)[np.unique(filtered_leaf_mask)] > 0)) if filtered_leaf_mask.size > 0 else 0
                    except Exception:
                        active_leaf = len([obj for obj in self.analysis_results.get('objects', []) if obj.get('id', 0) not in self._deleted_objects])
                active_scale = 0
                if self._current_scale_labels is not None:
                    try:
                        filtered_scale_mask = self._create_filtered_scale_mask()
                        active_scale = int(np.max(self._current_scale_labels[filtered_scale_mask])) if filtered_scale_mask.size > 0 else 0
                        # 위 한 줄은 라벨 맵 기준 최대 ID일 뿐이므로, 실제 활성 개수를 다시 계산
                        active_scale = len([sid for sid in np.unique(self._current_scale_labels) if sid > 0 and sid not in self._deleted_scale_objects])
                    except Exception:
                        active_scale = len([sid for sid in np.unique(self._current_scale_labels) if sid > 0 and sid not in self._deleted_scale_objects])
                total_deleted = len(self._deleted_objects) + len(self._deleted_scale_objects)
                messagebox.showinfo(
                    "객체 선택",
                    f"{type_name} 객체 #{object_id}가 {action}되었습니다.\n\n"
                    f"현재 표시중: Leaf {active_leaf}개, Scale {active_scale}개\n"
                    f"현재 삭제된 객체: Leaf {len(self._deleted_objects)}개, Scale {len(self._deleted_scale_objects)}개\n"
                    f"Ctrl+클릭으로 객체를 삭제/복원할 수 있습니다."
                )
                self._safe_refocus()
                
                # 미리보기 업데이트
                self.refresh_display_with_deletions()
                
            else:
                messagebox.showinfo("정보", "이 위치에는 객체가 없습니다.")
                self._safe_refocus()
                
        except Exception as e:
            print(f"객체 삭제 클릭 처리 오류: {e}")

    def clear_current_seeds(self):
        """현재 선택된 클래스의 시드 초기화"""
        current_mode = self.seed_mode.get() if CTK_AVAILABLE else "leaf"
        self._deactivate_delete_mode()
        self.seed_manager.clear_seeds(current_mode)
        
        # 마스크 캐시 무효화 (시드 초기화)
        self._invalidate_mask_cache()
        
        # 슈퍼픽셀 세그먼트 ID도 초기화
        if current_mode in self.seed_segment_ids:
            self.seed_segment_ids[current_mode].clear()
            print(f"{current_mode} 시드와 세그먼트 ID 초기화")
        
        self.update_display_image()

    def undo_last_seed(self):
        """마지막 시드 제거"""
        self.on_canvas_right_click(None)
    
    # 색상 모델 구축 기능 제거됨 (GrabCut 사용)

    def _deactivate_delete_mode(self):
        """시드 변경 시 삭제 모드 강제 해제"""
        if getattr(self, 'delete_mode_enabled', False) or getattr(self, 'delete_selected', None):
            self.delete_mode_enabled = False
            self.delete_selected = set()

    def _preview_split_result(self):
        try:
            mask = self._extract_selected_object_mask()
            if mask is None or np.sum(mask) == 0:
                messagebox.showwarning("경고", "선택한 객체 마스크를 찾을 수 없습니다.")
                self._safe_refocus()
                return
            if len(self.split_mode_points) < 2:
                return
            # 소영역 크롭
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            pad = 10
            y0 = max(0, y0 - pad); y1 = min(mask.shape[0]-1, y1 + pad)
            x0 = max(0, x0 - pad); x1 = min(mask.shape[1]-1, x1 + pad)
            roi_mask = mask[y0:y1+1, x0:x1+1].astype(np.uint8)
    
            # 워터셰드 마커: 두 점을 ROI 좌표로 변환
            (xA, yA), (xB, yB) = self.split_mode_points[:2]
            yA -= y0; yB -= y0; xA -= x0; xB -= x0
            markers = np.zeros_like(roi_mask, dtype=np.int32)
            if 0 <= yA < markers.shape[0] and 0 <= xA < markers.shape[1]:
                markers[yA, xA] = 1
            if 0 <= yB < markers.shape[0] and 0 <= xB < markers.shape[1]:
                markers[yB, xB] = 2
    
            # 거리변환 기반 워터셰드
            dist = ndimage.distance_transform_edt(roi_mask > 0)
            seg = watershed(-dist, markers=markers, mask=(roi_mask > 0))
            comp1 = (seg == 1)
            comp2 = (seg == 2)
            if comp1.sum() == 0 or comp2.sum() == 0:
                messagebox.showwarning("경고", "두 컴포넌트로 분리되지 않았습니다. 시드를 다시 지정하세요.")
                self._safe_refocus()
                return
    
            # 미리보기: 디스플레이에 2색으로 표시
            overlay = self.display_image.copy()
            dh, dw = overlay.shape[:2]
            # 전체 마스크 리사이즈 준비
            full1 = np.zeros_like(mask, dtype=np.uint8)
            full2 = np.zeros_like(mask, dtype=np.uint8)
            full1[y0:y1+1, x0:x1+1] = comp1.astype(np.uint8)
            full2[y0:y1+1, x0:x1+1] = comp2.astype(np.uint8)
            disp1 = cv2.resize(full1, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            disp2 = cv2.resize(full2, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            # 고대비 색상으로 확실히 구분되도록 직접 착색 (배경과 혼색 없이 선명 표시)
            overlay_f = overlay.astype(np.float32)
            col1 = np.array([0, 255, 0], dtype=np.float32)     # 선명한 녹색
            col2 = np.array([255, 0, 255], dtype=np.float32)   # 선명한 마젠타
            overlay_f[disp1] = col1
            overlay_f[disp2] = col2
            # 외곽선(고대비)로 다시 강조
            try:
                c1, _ = cv2.findContours(disp1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c2, _ = cv2.findContours(disp2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_f, c1, -1, (255, 255, 0), 1)
                cv2.drawContours(overlay_f, c2, -1, (0, 128, 255), 1)
            except Exception:
                pass
            overlay = (overlay_f * 0.92).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(overlay.astype(np.uint8))
            self.photo = ImageTk.PhotoImage(pil)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
            messagebox.showinfo("미리보기", "분리 미리보기를 표시했습니다. '완료'를 눌러 적용하세요.")
            self._safe_refocus()
            # 임시 보관
            self._split_preview = (y0, y1, x0, x1, comp1, comp2)
        except Exception as e:
            messagebox.showerror("오류", f"분리 미리보기 중 오류:\n{e}")
            self._safe_refocus()

    def _preview_merge_result(self):
        try:
            if not getattr(self, 'merge_mode_enabled', False):
                return
            if len(self.merge_selected) < 2:
                # 선택 강조만: 굵은 윤곽선 표시를 위해 결과 오버레이 호출
                self.show_result_overlay()
                return
            # 동일 클래스 집합 확인
            types = {t for (t, _) in self.merge_selected}
            if len(types) != 1:
                messagebox.showwarning("경고", "Leaf와 Scale을 함께 병합할 수 없습니다.")
                self._safe_refocus()
                return
            typ = list(types)[0]
            if typ == 'leaf' and self._current_instance_labels is None:
                return
            if typ == 'scale' and self._current_scale_labels is None:
                return
            labels = self._current_instance_labels if typ == 'leaf' else self._current_scale_labels
            merged_mask = np.zeros_like(labels, dtype=bool)
            for (_, oid) in self.merge_selected:
                merged_mask |= (labels == int(oid))
            if np.sum(merged_mask) == 0:
                self.show_result_overlay()
                return
            # 디스플레이 크기로 리사이즈
            if not hasattr(self, 'display_image') or self.display_image is None:
                self.update_display_image()
            base = self.display_image.copy()
            H, W = base.shape[:2]
            disp_mask = cv2.resize(merged_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            # 고대비 단일 색으로 채우기 + 굵은 윤곽
            color = np.array([255, 255, 0], dtype=np.float32)  # 노란색
            alpha = 0.5
            overlay = base.astype(np.float32)
            overlay[disp_mask] = overlay[disp_mask] * (1 - alpha) + color * alpha
            try:
                cnts, _ = cv2.findContours(disp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                thickness = int(self.settings.get("overlay_contour_thickness", 1)) + 2
                cv2.drawContours(overlay, cnts, -1, (255, 255, 255), thickness)
            except Exception:
                pass
            img = overlay.clip(0,255).astype(np.uint8)
            pil = Image.fromarray(img)
            self.photo = ImageTk.PhotoImage(pil)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
        except Exception as e:
            try:
                messagebox.showerror("오류", f"병합 미리보기 중 오류:\n{e}")
                self._safe_refocus()
            except Exception:
                pass
