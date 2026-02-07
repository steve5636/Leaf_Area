#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Operations for Advanced Leaf Analyzer
객체 분리/병합/삭제 기능
"""

import numpy as np
import cv2
from typing import Optional, Set
from tkinter import messagebox
from scipy import ndimage
from PIL import Image, ImageTk
from ..core.morphology import MorphologicalAnalyzer

try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    import tkinter as tk
    ctk = tk
    CTK_AVAILABLE = False


class ObjectOperations:
    """객체 편집 기능 믹스인 클래스"""




















    def _can_enter_split_mode(self) -> bool:
        if not self.analysis_results:
            return False
        leaf_cnt = len(self.analysis_results.get('objects', []))
        scale_cnt = 0
        if self._current_scale_labels is not None:
            scale_cnt = len([sid for sid in np.unique(self._current_scale_labels) if sid > 0])
        elif self.analysis_results.get('scale_mask') is not None:
            scale_cnt = int(np.sum(self.analysis_results['scale_mask']) > 0)
        return (leaf_cnt + scale_cnt) > 0

    def toggle_split_mode(self):
        if not self._can_enter_split_mode():
            messagebox.showwarning("경고", "분석 후에 사용해주세요. (객체가 0개)")
            self._safe_refocus()
            return
        self.split_mode_enabled = not self.split_mode_enabled
        self.split_mode_points = []
        self.split_selected_object = None
        if self.split_mode_enabled:
            # 다른 모드 비활성화
            if getattr(self, 'merge_mode_enabled', False):
                self.merge_mode_enabled = False
                self.merge_selected = set()
                self._merge_snapshot = None
            if getattr(self, 'delete_mode_enabled', False):
                self.delete_mode_enabled = False
                self.delete_selected = set()
            # 스냅샷 저장 (deepcopy로 numpy 보존)
            try:
                labels_snapshot = None if self._current_instance_labels is None else self._current_instance_labels.copy()
            except Exception:
                labels_snapshot = None
            try:
                analysis_snapshot = deepcopy(self.analysis_results)
            except Exception:
                analysis_snapshot = self.analysis_results
            self._split_snapshot = (labels_snapshot, analysis_snapshot)
            messagebox.showinfo("분리 모드", "객체를 클릭해 선택한 뒤, 분리할 두 지점을 찍어주세요.")
            self._safe_refocus()
        else:
            self._split_snapshot = None
        self.show_result_overlay()

    def toggle_merge_mode(self):
        if not self._can_enter_split_mode():
            messagebox.showwarning("경고", "분석 후에 사용해주세요. (객체가 0개)")
            self._safe_refocus()
            return
        self.merge_mode_enabled = not self.merge_mode_enabled
        self.merge_selected = set()
        if self.merge_mode_enabled:
            # 다른 모드 비활성화
            if getattr(self, 'split_mode_enabled', False):
                self.split_mode_enabled = False
                self.split_mode_points = []
                self.split_selected_object = None
                self._split_snapshot = None
            if getattr(self, 'delete_mode_enabled', False):
                self.delete_mode_enabled = False
                self.delete_selected = set()
            # 스냅샷 저장
            try:
                inst_snapshot = None if self._current_instance_labels is None else self._current_instance_labels.copy()
            except Exception:
                inst_snapshot = None
            try:
                scale_snapshot = None if self._current_scale_labels is None else self._current_scale_labels.copy()
            except Exception:
                scale_snapshot = None
            try:
                analysis_snapshot = deepcopy(self.analysis_results)
            except Exception:
                analysis_snapshot = self.analysis_results
            self._merge_snapshot = (inst_snapshot, scale_snapshot, analysis_snapshot)
            messagebox.showinfo("병합 모드", "병합할 객체를 2개 이상 선택하세요. 완료를 누르면 병합됩니다.")
            self._safe_refocus()
        else:
            self._merge_snapshot = None
        self.show_result_overlay()

    def toggle_delete_mode(self):
        """삭제 모드 토글: 다중 선택 후 일괄 삭제"""
        if not self._can_enter_split_mode():
            messagebox.showwarning("경고", "분석 후에 사용해주세요. (객체가 0개)")
            self._safe_refocus()
            return
        self.delete_mode_enabled = not self.delete_mode_enabled
        self.delete_selected = set()
        if self.delete_mode_enabled:
            # 다른 모드 비활성화
            if getattr(self, 'split_mode_enabled', False):
                self.split_mode_enabled = False
                self.split_mode_points = []
                self.split_selected_object = None
                self._split_snapshot = None
            if getattr(self, 'merge_mode_enabled', False):
                self.merge_mode_enabled = False
                self.merge_selected = set()
                self._merge_snapshot = None
            messagebox.showinfo("삭제 모드", "삭제할 객체를 클릭해 선택한 뒤 '선택 삭제'를 눌러주세요.\nCtrl+클릭 복원은 그대로 유지됩니다.")
            self._safe_refocus()
        self.show_result_overlay()

    def delete_apply(self):
        """선택된 객체 일괄 삭제"""
        if not self.analysis_results:
            messagebox.showwarning("경고", "먼저 분석을 실행하세요.")
            self._safe_refocus()
            return
        if not getattr(self, 'delete_selected', set()):
            messagebox.showinfo("정보", "선택된 객체가 없습니다.")
            self._safe_refocus()
            return
        leaf_added = 0
        scale_added = 0
        for obj_type, obj_id in list(self.delete_selected):
            if obj_type == "leaf":
                if int(obj_id) not in self._deleted_objects:
                    self._deleted_objects.add(int(obj_id))
                    leaf_added += 1
            else:
                if int(obj_id) not in self._deleted_scale_objects:
                    self._deleted_scale_objects.add(int(obj_id))
                    scale_added += 1
        self.delete_selected = set()
        self.refresh_display_with_deletions()
        messagebox.showinfo("삭제 완료", f"삭제됨: Leaf {leaf_added}개, Scale {scale_added}개\nCtrl+클릭으로 복원 가능합니다.")
        self._safe_refocus()

    def delete_clear(self):
        """삭제 선택 해제"""
        self.delete_selected = set()
        self.show_result_overlay()
    
    def _show_split_overlay_highlight(self):
        # 현재 오버레이를 재표시(불투명도 강화는 show_result_overlay 내에서 반영됨)
        try:
            self.show_result_overlay()
        except Exception:
            pass

    def _extract_selected_object_mask(self) -> Optional[np.ndarray]:
        if self.split_selected_object is None:
            return None
        typ, oid = self.split_selected_object
        if typ == 'leaf':
            if self._current_instance_labels is None:
                return None
            return (self._current_instance_labels == int(oid))
        else:
            if self._current_scale_labels is None:
                return None
            return (self._current_scale_labels == int(oid))

    def _preview_split_result(self):
        return super()._preview_split_result()
    def _preview_merge_result(self):
        return super()._preview_merge_result()
    def split_apply(self):
        # 병합 모드면 병합 적용으로 라우팅
        if getattr(self, 'merge_mode_enabled', False):
            self._merge_apply()
            return
        if not getattr(self, 'split_mode_enabled', False):
            return
        try:
            if not hasattr(self, '_split_preview'):
                messagebox.showwarning("경고", "먼저 시드를 두 점 찍어 미리보기를 생성하세요.")
                self._safe_refocus()
                return
            y0, y1, x0, x1, comp1, comp2 = self._split_preview
            mask = self._extract_selected_object_mask()
            if mask is None:
                return
            # 기존 객체 ID와 타입
            typ, oid = self.split_selected_object
            oid = int(oid)

            # 인스턴스 라벨 업데이트 (Leaf 우선)
            if typ == 'leaf':
                labels = self._current_instance_labels.copy()
                labels[labels == oid] = 0
                new_id1 = int(labels.max()) + 1
                labels[y0:y1+1, x0:x1+1][comp1] = new_id1
                new_id2 = new_id1 + 1
                labels[y0:y1+1, x0:x1+1][comp2] = new_id2
                self._current_instance_labels = labels

                # analysis_results.objects 갱신
                objs = [o for o in self.analysis_results.get('objects', []) if int(o.get('id', -1)) != oid]
                def _mk_obj(bin_mask, new_id):
                    cnts, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not cnts:
                        return None
                    c = max(cnts, key=cv2.contourArea)
                    data = MorphologicalAnalyzer.analyze_contour(c)
                    data['id'] = new_id
                    data['contour'] = c
                    return data
                full1 = np.zeros_like(mask, dtype=np.uint8); full1[y0:y1+1, x0:x1+1] = comp1.astype(np.uint8)
                full2 = np.zeros_like(mask, dtype=np.uint8); full2[y0:y1+1, x0:x1+1] = comp2.astype(np.uint8)
                objA = _mk_obj(full1>0, new_id1)
                objB = _mk_obj(full2>0, new_id2)
                if objA: objs.append(objA)
                if objB: objs.append(objB)
                self.analysis_results['objects'] = objs
            else:
                # Scale 라벨맵 분할
                labels = self._current_scale_labels.copy()
                labels[labels == oid] = 0
                new_id1 = int(labels.max()) + 1
                labels[y0:y1+1, x0:x1+1][comp1] = new_id1
                new_id2 = new_id1 + 1
                labels[y0:y1+1, x0:x1+1][comp2] = new_id2
                self._current_scale_labels = labels

            # 모드 종료와 표시 갱신
            self.split_mode_enabled = False
            self.split_mode_points = []
            self.split_selected_object = None
            self._split_snapshot = None
            if hasattr(self, '_split_preview'):
                del self._split_preview
            self.show_result_overlay()
            messagebox.showinfo("성공", "분리가 완료되었습니다.")
            self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"분리 적용 중 오류:\n{e}")
            self._safe_refocus()

    def _merge_apply(self):
        try:
            if not getattr(self, 'merge_mode_enabled', False):
                return
            if len(self.merge_selected) < 2:
                messagebox.showwarning("경고", "두 객체 이상을 선택하세요.")
                self._safe_refocus()
                return
            types = {t for (t, _) in self.merge_selected}
            if len(types) != 1:
                messagebox.showwarning("경고", "Leaf와 Scale을 함께 병합할 수 없습니다.")
                self._safe_refocus()
                return
            typ = list(types)[0]
            if typ == 'leaf':
                if self._current_instance_labels is None:
                    messagebox.showwarning("경고", "병합할 Leaf 라벨이 없습니다.")
                    self._safe_refocus()
                    return
                labels = self._current_instance_labels.copy()
                merge_ids = sorted(int(oid) for (_, oid) in self.merge_selected)
                merged_mask = np.zeros_like(labels, dtype=bool)
                for oid in merge_ids:
                    merged_mask |= (labels == oid)
                if np.sum(merged_mask) == 0:
                    messagebox.showwarning("경고", "병합 결과가 비어있습니다.")
                    self._safe_refocus()
                    return
                # 기존 ID 제거
                for oid in merge_ids:
                    labels[labels == oid] = 0
                new_id = int(labels.max()) + 1
                labels[merged_mask] = new_id
                self._current_instance_labels = labels
                # objects 갱신
                objs = [o for o in self.analysis_results.get('objects', []) if int(o.get('id', -1)) not in set(merge_ids)]
                data = MorphologicalAnalyzer.analyze_mask_with_holes(merged_mask)
                data['id'] = int(new_id)
                objs.append(data)
                self.analysis_results['objects'] = objs
                # 삭제 집합 정리
                for oid in merge_ids:
                    if oid in self._deleted_objects:
                        self._deleted_objects.discard(oid)
            else:  # scale
                if self._current_scale_labels is None:
                    messagebox.showwarning("경고", "병합할 Scale 라벨이 없습니다.")
                    self._safe_refocus()
                    return
                labels = self._current_scale_labels.copy()
                merge_ids = sorted(int(oid) for (_, oid) in self.merge_selected)
                merged_mask = np.zeros_like(labels, dtype=bool)
                for oid in merge_ids:
                    merged_mask |= (labels == oid)
                if np.sum(merged_mask) == 0:
                    messagebox.showwarning("경고", "병합 결과가 비어있습니다.")
                    self._safe_refocus()
                    return
                for oid in merge_ids:
                    labels[labels == oid] = 0
                new_id = int(labels.max()) + 1
                labels[merged_mask] = new_id
                self._current_scale_labels = labels
                for oid in merge_ids:
                    if oid in self._deleted_scale_objects:
                        self._deleted_scale_objects.discard(oid)
            # 종료 및 표시 갱신
            self.merge_mode_enabled = False
            self.merge_selected = set()
            self._merge_snapshot = None
            self.show_result_overlay()
            messagebox.showinfo("성공", "병합이 완료되었습니다.")
            self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"병합 적용 중 오류:\n{e}")
            self._safe_refocus()

    def _merge_undo(self):
        try:
            if not self._merge_snapshot:
                messagebox.showinfo("정보", "되돌릴 항목이 없습니다.")
                self._safe_refocus()
                return
            inst_snapshot, scale_snapshot, analysis_snapshot = self._merge_snapshot
            if inst_snapshot is not None:
                self._current_instance_labels = inst_snapshot
            if scale_snapshot is not None:
                self._current_scale_labels = scale_snapshot
            if analysis_snapshot is not None:
                self.analysis_results = analysis_snapshot
            self.merge_mode_enabled = False
            self.merge_selected = set()
            self._merge_snapshot = None
            self.show_result_overlay()
            messagebox.showinfo("복구", "병합 작업이 되돌려졌습니다.")
            self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"되돌리기 중 오류:\n{e}")
            self._safe_refocus()

    def split_undo(self):
        try:
            # 병합 모드면 병합 되돌리기로 라우팅
            if getattr(self, 'merge_mode_enabled', False) or self._merge_snapshot:
                self._merge_undo()
                return
            if not self._split_snapshot:
                messagebox.showinfo("정보", "되돌릴 항목이 없습니다.")
                self._safe_refocus()
                return
            labels_snapshot, analysis_snapshot = self._split_snapshot
            if labels_snapshot is not None:
                self._current_instance_labels = labels_snapshot
            if analysis_snapshot is not None:
                self.analysis_results = analysis_snapshot
                # 복원된 마스크가 리스트일 경우 numpy bool로 강제 변환
                try:
                    if isinstance(self.analysis_results.get('leaf_mask', None), list):
                        self.analysis_results['leaf_mask'] = np.asarray(self.analysis_results['leaf_mask']).astype(bool)
                except Exception:
                    pass
                try:
                    if isinstance(self.analysis_results.get('scale_mask', None), list):
                        self.analysis_results['scale_mask'] = np.asarray(self.analysis_results['scale_mask']).astype(bool)
                except Exception:
                    pass
            self.split_mode_points = []
            self.split_selected_object = None
            self.show_result_overlay()
            messagebox.showinfo("복구", "분리 작업이 되돌려졌습니다.")
            self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"되돌리기 중 오류:\n{e}")
            self._safe_refocus()

    # 병합 모드에서 기존 버튼 재사용: 분리의 완료/되돌리기 버튼과 동일 UI를 공유
    # 완료 버튼이 눌릴 때 병합 모드면 병합 적용, 분리 모드면 기존 분리 적용

    def reset_object_deletions(self):
        """삭제된 객체 모두 복원 (Leaf와 Scale 모두)"""
        deleted_leaf_count = len(self._deleted_objects)
        deleted_scale_count = len(self._deleted_scale_objects)
        total_deleted = deleted_leaf_count + deleted_scale_count
        
        if total_deleted > 0:
            self._deleted_objects.clear()
            self._deleted_scale_objects.clear()
            self.refresh_display_with_deletions()
            messagebox.showinfo(
                "복원 완료", 
                f"총 {total_deleted}개의 객체가 모두 복원되었습니다.\n"
                f"(Leaf: {deleted_leaf_count}개, Scale: {deleted_scale_count}개)"
            )
            self._safe_refocus()
        else:
            messagebox.showinfo("정보", "삭제된 객체가 없습니다.")
            self._safe_refocus()
    
    def _find_object_at_position(self, x: int, y: int, include_deleted: bool = False) -> Optional[tuple]:
        """클릭 위치에서 객체 찾기 (Leaf와 Scale 모두 확인)
        include_deleted=True 이면 삭제된 객체도 히트 대상으로 포함하여 복원 클릭을 지원."""
        # 동적 허용 반경(px, 원본 좌표 기준): 디스플레이 스케일이 작을수록 반경 확대
        try:
            ds = float(getattr(self, 'display_scale', 1.0))
        except Exception:
            ds = 1.0
        hit_r = max(1, int(round(6.0 / max(ds, 1e-6))))

        # Leaf 먼저: 주변 창에서 최빈(최다 픽셀) ID를 선택
        if self._current_instance_labels is not None:
            try:
                h, w = self._current_instance_labels.shape
                if 0 <= y < h and 0 <= x < w:
                    x0, x1 = max(0, x - hit_r), min(w - 1, x + hit_r)
                    y0, y1 = max(0, y - hit_r), min(h - 1, y + hit_r)
                    patch = self._current_instance_labels[y0:y1+1, x0:x1+1]
                    if patch.size > 0:
                        vals, counts = np.unique(patch, return_counts=True)
                        # 배경과 삭제된 객체 제외
                        valid = []
                        for vid, cnt in zip(vals, counts):
                            if vid <= 0:
                                continue
                            if (not include_deleted) and (vid in getattr(self, '_deleted_objects', set())):
                                continue
                            valid.append((int(vid), int(cnt)))
                        if valid:
                            # 최다 픽셀 ID 선택
                            best_id = max(valid, key=lambda t: t[1])[0]
                            return ("leaf", best_id)
            except Exception as e:
                print(f"Leaf 객체 찾기 오류: {e}")

        # Scale: 동일 로직
        if self._current_scale_labels is not None:
            try:
                h, w = self._current_scale_labels.shape
                if 0 <= y < h and 0 <= x < w:
                    x0, x1 = max(0, x - hit_r), min(w - 1, x + hit_r)
                    y0, y1 = max(0, y - hit_r), min(h - 1, y + hit_r)
                    patch = self._current_scale_labels[y0:y1+1, x0:x1+1]
                    if patch.size > 0:
                        vals, counts = np.unique(patch, return_counts=True)
                        valid = []
                        for vid, cnt in zip(vals, counts):
                            if vid <= 0:
                                continue
                            if (not include_deleted) and (vid in getattr(self, '_deleted_scale_objects', set())):
                                continue
                            valid.append((int(vid), int(cnt)))
                        if valid:
                            best_id = max(valid, key=lambda t: t[1])[0]
                            return ("scale", best_id)
            except Exception as e:
                print(f"Scale 객체 찾기 오류: {e}")

        return None
    
    def refresh_display_with_deletions(self):
        """삭제된 객체를 제외한 미리보기 업데이트"""
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            return
            
        try:
            # 최종 결과 오버레이 경로 재사용(디스플레이 크기에서 그리기 → 두께/폰트 일관)
            self.show_result_overlay()
            print("마스크 업데이트: 삭제 상태 반영 완료 (결과 오버레이 경로)")
            
        except Exception as e:
            print(f"미리보기 업데이트 실패: {e}")
    
    def _create_filtered_mask(self) -> np.ndarray:
        """삭제된 객체를 제외한 마스크 생성"""
        if self._current_instance_labels is None:
            print("_create_filtered_mask: _current_instance_labels가 None - 빈 마스크 반환")
            return np.zeros((0, 0), dtype=bool)
            
        # 삭제되지 않은 객체만 포함
        labels = self._current_instance_labels
        # 허용 ID 집합 구성
        if hasattr(self, 'analysis_results') and self.analysis_results and 'objects' in self.analysis_results:
            try:
                allowed_ids = np.array([int(o.get('id', 0)) for o in self.analysis_results['objects']], dtype=np.int32)
            except Exception:
                allowed_ids = np.array([], dtype=np.int32)
        else:
            allowed_ids = np.array([], dtype=np.int32)
        if allowed_ids.size == 0:
            print("_create_filtered_mask: 활성 객체 0개 (allowed_ids 비어있음)")
            return np.zeros_like(labels, dtype=bool)
        # 삭제된 ID 제거
        if hasattr(self, '_deleted_objects') and len(self._deleted_objects) > 0:
            deleted_ids = np.array(sorted(list(self._deleted_objects)), dtype=np.int32)
            allowed_ids = allowed_ids[~np.isin(allowed_ids, deleted_ids)]
        if allowed_ids.size == 0:
            print("_create_filtered_mask: 활성 객체 0개 (모두 삭제됨)")
            return np.zeros_like(labels, dtype=bool)
        filtered_mask = np.isin(labels, allowed_ids)
        print(f"_create_filtered_mask: {int(np.unique(labels[filtered_mask]).size)}개 활성 객체, {int(filtered_mask.sum())}픽셀 생성")
        return filtered_mask
    
    def _create_filtered_scale_mask(self) -> np.ndarray:
        """삭제된 Scale 객체를 제외한 Scale 마스크 생성"""
        if self._current_scale_labels is None:
            print("_create_filtered_scale_mask: _current_scale_labels가 None - 빈 마스크 반환")
            return np.zeros((0, 0), dtype=bool)
            
        labels = self._current_scale_labels
        unique_ids = np.unique(labels)
        keep_ids = unique_ids[(unique_ids > 0)]
        if hasattr(self, '_deleted_scale_objects') and len(self._deleted_scale_objects) > 0:
            deleted_ids = np.array(sorted(list(self._deleted_scale_objects)), dtype=np.int32)
            keep_ids = keep_ids[~np.isin(keep_ids, deleted_ids)]
        if keep_ids.size == 0:
            print("_create_filtered_scale_mask: 0개 활성 Scale 객체")
            return np.zeros_like(labels, dtype=bool)
        filtered_scale_mask = np.isin(labels, keep_ids)
        print(f"_create_filtered_scale_mask: {int(keep_ids.size)}개 활성 Scale 객체, {int(filtered_scale_mask.sum())}픽셀 생성")
        return filtered_scale_mask

