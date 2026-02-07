#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Exporters for Advanced Leaf Analyzer
CSV/JSON/YOLO/COCO 내보내기
"""

import os
import json
import csv
from typing import List, Tuple

import cv2
import numpy as np
from tkinter import filedialog, messagebox


class DataExporter:
    """데이터 내보내기 믹스인 클래스"""

    def _compute_export_pixels_per_cm2(self) -> float:
        """내보내기 시점의 활성 Scale 객체를 기반으로 pixels_per_cm2 재계산.
        활성 Scale이 없으면 분석 시 계산값을 사용하고, 그것도 없으면 1을 반환.
        """
        try:
            # 스케일 면적(cm^2) - UI 설정값 사용 (기본 4.0)
            scale_area_cm2 = float(self.settings.get("scale_area_cm2", 4.0))
            # 활성 Scale 마스크 계산
            if self._current_scale_labels is not None:
                active_scale_mask = self._create_filtered_scale_mask()
                if active_scale_mask is not None and active_scale_mask.size > 0:
                    area_px = int(np.sum(active_scale_mask))
                    if area_px > 0:
                        return float(area_px) / float(scale_area_cm2)
            # 라벨맵이 없으면 분석 결과의 scale_mask 사용
            if hasattr(self, 'analysis_results') and self.analysis_results is not None:
                sm = self.analysis_results.get('scale_mask')
                if sm is not None:
                    area_px = int(np.sum(sm))
                    if area_px > 0:
                        return float(area_px) / float(scale_area_cm2)
                # 분석 시 계산된 값을 폴백
                ppcm2 = float(self.analysis_results.get('pixels_per_cm2', 1.0))
                return ppcm2 if ppcm2 > 0 else 1.0
        except Exception:
            pass
        return 1.0

    def _get_export_scale_factor(self) -> tuple[float | None, bool]:
        """(pixels_per_cm2, has_scale) 반환. has_scale=False면 cm 변환은 비움."""
        try:
            # 활성 Scale 존재 여부 판단
            has_scale = False
            if self._current_scale_labels is not None:
                ids = [sid for sid in np.unique(self._current_scale_labels) if sid > 0]
                active_ids = [sid for sid in ids if sid not in getattr(self, '_deleted_scale_objects', set())]
                has_scale = len(active_ids) > 0
            elif self.analysis_results and self.analysis_results.get('scale_mask') is not None:
                has_scale = np.sum(self.analysis_results['scale_mask']) > 0
            if not has_scale:
                return (None, False)
            return (self._compute_export_pixels_per_cm2(), True)
        except Exception:
            return (None, False)

    def _json_safe(self, obj):
        """JSON 직렬화 안전 변환: numpy 타입/ndarray를 파이썬 기본형/리스트로 변환"""
        try:
            import numpy as _np
        except Exception:
            _np = None
        # numpy scalar
        if _np is not None and isinstance(obj, _np.generic):
            return obj.item()
        # numpy array
        if _np is not None and isinstance(obj, _np.ndarray):
            return obj.tolist()
        # list/tuple
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(x) for x in obj]
        # dict
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        return obj

    def _gather_active_leaf_objects(self):
        """활성 Leaf 객체 리스트 반환"""
        if not self.analysis_results:
            return []
        active = [
            obj for obj in self.analysis_results.get('objects', [])
            if obj.get('id', 0) not in self._deleted_objects
        ]
        return active

    def _gather_active_scale_masks(self):
        """활성 Scale 마스크들의 리스트(bool mask)와 각각의 연결성 라벨 ID 반환"""
        masks = []
        if self._current_scale_labels is not None:
            labels = self._current_scale_labels
            for sid in np.unique(labels):
                if sid > 0 and sid not in self._deleted_scale_objects:
                    masks.append((sid, labels == sid))
        elif self.analysis_results and self.analysis_results.get('scale_mask') is not None:
            masks.append((1, self.analysis_results['scale_mask']))
        return masks

    def _ensure_dir(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    def export_yolo_obb(self):
        """Ultralytics YOLO OBB 형식(4 코너 정규화)으로 활성 Leaf/Scale 내보내기
        참조: https://docs.ultralytics.com/ko/datasets/obb/
        """
        if self.original_image is None or not self.analysis_results:
            messagebox.showerror("오류", "먼저 분석을 실행해주세요.")
            self._safe_refocus()
            return
        try:
            # 저장 디렉토리 선택
            out_dir = filedialog.askdirectory(title="YOLO OBB 내보낼 디렉토리 선택")
            self._safe_refocus()
            if not out_dir:
                return
            self._ensure_dir(out_dir)
            images_dir = os.path.join(out_dir, "images")
            labels_dir = os.path.join(out_dir, "labels")
            self._ensure_dir(images_dir)
            self._ensure_dir(labels_dir)

            # 이미지 저장 (원본 복사)
            img = self.original_image
            h, w = img.shape[:2]
            # 파일명 결정
            base_name = os.path.splitext(os.path.basename(getattr(self, 'current_image_path', 'image')))[0]
            img_path = os.path.join(images_dir, base_name + ".jpg")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 활성 객체 수집
            active_leaf = self._gather_active_leaf_objects()
            active_scales = self._gather_active_scale_masks()

            # 라벨 작성
            label_lines = []
            yolo_items = []  # (class_id, coords_norm[8], object_id)
            # 클래스 ID: Leaf=0, Scale=2
            leaf_cid = 0
            scale_cid = 2
            # Leaf: contour/bbox → 강건 OBB 추출기 사용 (bbox 4점은 그대로 OBB로 허용)
            for obj in active_leaf:
                if 'contour' in obj:
                    cnt = np.asarray(obj['contour'], dtype=np.float32).reshape(-1, 2)
                    quad = self._robust_obb_from_points(cnt, w, h)
                elif 'bounding_box' in obj:
                    bb = np.asarray(obj['bounding_box'], dtype=float).reshape(-1, 2)
                    # 경계 접촉이면 AABB, 아니면 제공된 4점(회전 사각형)을 그대로 사용
                    if self._touching_border(bb, w, h):
                        quad = self._aabb(bb)
                    else:
                        quad = bb
                else:
                    continue
                line = self._yolo_obb_line(leaf_cid, quad, w, h)
                label_lines.append(line)
                try:
                    obj_id = int(obj.get('id', -1))
                except Exception:
                    obj_id = -1
                coords_norm = list(map(float, line.split()[1:]))
                yolo_items.append((leaf_cid, coords_norm, obj_id))

            # Scale: 마스크 → 강건 OBB
            for sid, smask in active_scales:
                quad = self._scale_obb_from_mask(smask, w, h)
                if quad is None:
                    ys, xs = np.where(smask)
                    if len(xs) == 0:
                        continue
                    pts = np.column_stack([xs, ys]).astype(np.float32)
                    quad = self._robust_obb_from_points(pts, w, h)
                line = self._yolo_obb_line(scale_cid, quad, w, h)
                label_lines.append(line)
                coords_norm = list(map(float, line.split()[1:]))
                yolo_items.append((scale_cid, coords_norm, int(sid)))

            # 라벨 저장
            with open(os.path.join(labels_dir, base_name + ".txt"), 'w', encoding='utf-8') as f:
                for line in label_lines:
                    f.write(line + "\n")

            # classes.txt 저장
            # classes.txt: index 0과 2를 맞추기 위해 placeholder 포함
            with open(os.path.join(out_dir, "classes.txt"), 'w', encoding='utf-8') as f:
                f.write("leaf\nunused\nscale\n")

            # 디버깅 이미지 저장 (images_debug)
            try:
                debug_dir = os.path.join(out_dir, "images_debug")
                self._ensure_dir(debug_dir)
                dbg = img.copy()
                for cid, coords, oid in yolo_items:
                    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    pts_i = pts.astype(np.int32)
                    color = (0, 255, 0) if cid == leaf_cid else (255, 0, 0)
                    cv2.polylines(dbg, [pts_i], True, color, 2)
                    tx, ty = int(pts_i[0][0]), int(pts_i[0][1])
                    cv2.putText(dbg, f"{cid}:{oid}", (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imwrite(os.path.join(debug_dir, base_name + "_obb_viz.jpg"), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

            messagebox.showinfo("성공", f"YOLO OBB 내보내기 완료:\n{out_dir}\n라벨 {len(label_lines)}개")
            self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"YOLO OBB 내보내기 실패:\n{e}")
            self._safe_refocus()

    def export_yolo_seg(self):
        """Ultralytics YOLO Segmentation 형식으로 활성 Leaf/Scale 내보내기"""
        if self.original_image is None or not self.analysis_results:
            messagebox.showerror("오류", "먼저 분석을 실행해주세요.")
            self._safe_refocus()
            return
        try:
            out_dir = filedialog.askdirectory(title="YOLO Seg 내보낼 디렉토리 선택")
            self._safe_refocus()
            if not out_dir:
                return
            self._ensure_dir(out_dir)
            images_dir = os.path.join(out_dir, "images")
            labels_dir = os.path.join(out_dir, "labels")
            self._ensure_dir(images_dir)
            self._ensure_dir(labels_dir)

            # 이미지 저장 (원본 복사)
            img = self.original_image
            h, w = img.shape[:2]
            base_name = os.path.splitext(os.path.basename(getattr(self, 'current_image_path', 'image')))[0]
            img_path = os.path.join(images_dir, base_name + ".jpg")
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 활성 객체 수집
            active_leaf = self._gather_active_leaf_objects()
            active_scales = self._gather_active_scale_masks()

            # 라벨 작성
            seg_lines = []
            # 클래스 ID: Leaf=0, Scale=2
            leaf_cid = 0
            scale_cid = 2

            # Leaf: contour 기반 폴리곤
            for obj in active_leaf:
                if 'contour' not in obj:
                    continue
                cnt = np.asarray(obj['contour'], dtype=np.float32).reshape(-1, 2)
                if cnt.shape[0] < 3:
                    continue
                # normalize
                coords_norm = []
                for (x, y) in cnt:
                    coords_norm.append(float(x) / float(w))
                    coords_norm.append(float(y) / float(h))
                seg_lines.append((leaf_cid, coords_norm, int(obj.get('id', -1))))

            # Scale: 마스크 → 폴리곤
            for sid, smask in active_scales:
                polys = self._mask_to_polygon(smask)
                if not polys:
                    continue
                for poly in polys:
                    coords_norm = []
                    for k in range(0, len(poly), 2):
                        coords_norm.append(float(poly[k]) / float(w))
                        coords_norm.append(float(poly[k+1]) / float(h))
                    seg_lines.append((scale_cid, coords_norm, int(sid)))

            # 라벨 저장
            with open(os.path.join(labels_dir, base_name + ".txt"), 'w', encoding='utf-8') as f:
                for cid, coords, oid in seg_lines:
                    if len(coords) < 6:
                        continue
                    line = " ".join([str(cid)] + [f"{v:.6f}" for v in coords])
                    f.write(line + "\n")

            # classes.txt 저장
            with open(os.path.join(out_dir, "classes.txt"), 'w', encoding='utf-8') as f:
                f.write("leaf\nunused\nscale\n")

            # 디버깅 이미지 저장 (images_debug)
            try:
                debug_dir = os.path.join(out_dir, "images_debug")
                self._ensure_dir(debug_dir)
                try:
                    from PIL import Image as _PILImage
                    from PIL import ImageDraw as _PILDraw
                    pil_img = _PILImage.fromarray(img.copy()).convert("RGB")
                    draw = _PILDraw.Draw(pil_img)
                    for (cid, coords, oid) in seg_lines:
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            continue
                        pts = [(int(round(coords[k]*w)), int(round(coords[k+1]*h))) for k in range(0, len(coords), 2)]
                        col = (0, 255, 0) if cid == leaf_cid else (255, 0, 0)
                        draw.polygon(pts, outline=col)
                        x, y = pts[0]
                        draw.text((x+3, y+3), f"{cid}:{oid}", fill=col)
                    pil_img.save(os.path.join(debug_dir, base_name + "_seg_viz.jpg"))
                except Exception:
                    # fallback: OpenCV contour
                    canvas_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    for (cid, coords, oid) in seg_lines:
                        if len(coords) < 6 or len(coords) % 2 != 0:
                            continue
                        pts = np.array([(int(round(coords[k]*w)), int(round(coords[k+1]*h))) for k in range(0, len(coords), 2)], dtype=np.int32)
                        color = (0, 255, 0) if cid == leaf_cid else (0, 0, 255)
                        cv2.polylines(canvas_bgr, [pts], True, color, 2)
                    pil_img = _PILImage.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
                    pil_img.save(os.path.join(debug_dir, base_name + "_seg_viz.jpg"))
            except Exception:
                pass

            messagebox.showinfo("성공", f"YOLO Seg 내보내기 완료:\n{out_dir}\n라벨 {len(seg_lines)}개")
            self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"YOLO Seg 내보내기 실패:\n{e}")
            self._safe_refocus()

    def export_coco_seg(self):
        """COCO Segmentation(RLE) 포맷으로 활성 Leaf/Scale 내보내기"""
        if self.original_image is None or not self.analysis_results:
            messagebox.showerror("오류", "먼저 분석을 실행해주세요.")
            self._safe_refocus()
            return
        try:
            out_dir = filedialog.askdirectory(title="COCO Seg 내보낼 디렉토리 선택")
            self._safe_refocus()
            if not out_dir:
                return
            self._ensure_dir(out_dir)
            images_dir = os.path.join(out_dir, "images")
            self._ensure_dir(images_dir)

            img = self.original_image
            h, w = img.shape[:2]
            base_name = os.path.splitext(os.path.basename(getattr(self, 'current_image_path', 'image')))[0]
            img_file = base_name + ".jpg"
            cv2.imwrite(os.path.join(images_dir, img_file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 활성 객체 수집
            active_leaf = self._gather_active_leaf_objects()
            active_scales = self._gather_active_scale_masks()

            # COCO annotations.json 로드/초기화
            ann_path = os.path.join(out_dir, "annotations.json")
            coco = None
            if os.path.exists(ann_path):
                try:
                    with open(ann_path, 'r', encoding='utf-8') as f:
                        coco = json.load(f)
                except Exception:
                    coco = None
            if not coco:
                coco = {
                    "info": {"description": "Advanced Leaf Analyzer Export", "version": "1.0"},
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {"id": 0, "name": "leaf", "supercategory": "plant"},
                        {"id": 2, "name": "scale", "supercategory": "measurement"}
                    ]
                }
            # 카테고리 보장 (기존 파일에 없을 수 있음)
            if 'categories' not in coco or not coco['categories']:
                coco['categories'] = [
                    {"id": 0, "name": "leaf", "supercategory": "plant"}, 
                    {"id": 2, "name": "scale", "supercategory": "measurement"}
                ]
            # id 0 leaf, id 2 scale 카테고리가 없으면 추가 (supercategory 포함)
            try:
                cats = coco.get('categories', [])
                has_leaf0 = any(int(c.get('id', -1)) == 0 and str(c.get('name','')) == 'leaf' for c in cats)
                if not has_leaf0:
                    cats.append({"id": 0, "name": "leaf", "supercategory": "plant"})
                has_scale2 = any(int(c.get('id', -1)) == 2 and str(c.get('name','')) == 'scale' for c in cats)
                if not has_scale2:
                    cats.append({"id": 2, "name": "scale", "supercategory": "measurement"})
                # 기존 카테고리에 supercategory 추가 (없는 경우)
                for cat in cats:
                    if 'supercategory' not in cat:
                        if cat.get('name') == 'leaf':
                            cat['supercategory'] = 'plant'
                        elif cat.get('name') == 'scale':
                            cat['supercategory'] = 'measurement'
                        else:
                            cat['supercategory'] = 'object'
                coco['categories'] = cats
            except Exception:
                pass

            # 이미지 ID 할당 (동일 파일명이 이미 있으면 재사용)
            images_list = coco.get('images', [])
            existing_image_id = None
            for im in images_list:
                try:
                    if str(im.get('file_name', '')) == str(img_file):
                        existing_image_id = int(im.get('id', 0))
                        break
                except Exception:
                    continue
            if existing_image_id is None:
                try:
                    max_img_id = max([int(i.get('id', 0)) for i in images_list]) if images_list else 0
                except Exception:
                    max_img_id = len(images_list)
                image_id = int(max_img_id) + 1
                coco['images'].append({
                    "id": image_id,
                    "file_name": img_file,
                    "width": w,
                    "height": h
                })
            else:
                image_id = int(existing_image_id)

            # 동일 이미지에 대한 기존 주석 제거(중복 누적 방지)
            try:
                existing_anns = coco.get('annotations', [])
                coco['annotations'] = [a for a in existing_anns if int(a.get('image_id', -1)) != int(image_id)]
            except Exception:
                pass

            # annotation ID 시작점 계산
            try:
                ann_ids = [int(a.get('id', 0)) for a in coco.get('annotations', [])]
                ann_id = (max(ann_ids) + 1) if ann_ids else 1
            except Exception:
                ann_id = len(coco.get('annotations', [])) + 1

            # Polygon 형식 사용 여부 확인
            use_polygon = getattr(self, 'use_polygon_format', None)
            use_polygon = use_polygon.get() if use_polygon else False

            # Leaf: 라벨맵에서 마스크 복구
            for obj in active_leaf:
                try:
                    oid = int(obj.get('id', -1))
                except Exception:
                    oid = -1
                if oid <= 0 or self._current_instance_labels is None:
                    continue
                mask = (self._current_instance_labels == oid)
                if not np.any(mask):
                    continue

                # Polygon 또는 RLE 형식 선택
                if use_polygon:
                    segmentation = self._mask_to_polygon(mask)
                else:
                    segmentation = {"size": [int(h), int(w)], "counts": self._rle_counts_from_mask(mask)}

                bbox = self._bbox_from_mask(mask)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": int(image_id),
                    "category_id": 0,
                    "segmentation": segmentation,
                    "area": int(mask.sum()),
                    "bbox": [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)],
                    "iscrowd": 0,
                    "leaf_object_id": int(oid)
                })
                ann_id += 1

            # Scale: 라벨맵에서 마스크
            for sid, smask in active_scales:
                mask = smask.astype(bool)
                if not np.any(mask):
                    continue

                # Polygon 또는 RLE 형식 선택
                if use_polygon:
                    segmentation = self._mask_to_polygon(mask)
                else:
                    segmentation = {"size": [int(h), int(w)], "counts": self._rle_counts_from_mask(mask)}

                bbox = self._bbox_from_mask(mask)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": int(image_id),
                    "category_id": 2,
                    "segmentation": segmentation,
                    "area": int(mask.sum()),
                    "bbox": [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)],
                    "iscrowd": 0,
                    "scale_object_id": int(sid)
                })
                ann_id += 1

            # 저장 (기존 annotations.json에 병합 반영)
            with open(ann_path, 'w', encoding='utf-8') as f:
                json.dump(coco, f, ensure_ascii=False, indent=2)

            # === 디버그 시각화 ===
            try:
                debug_dir = os.path.join(out_dir, "images_debug")
                self._ensure_dir(debug_dir)

                # 원본 이미지 복사
                vis_img = img.copy()

                # 이 이미지의 주석만 필터링
                current_anns = [ann for ann in coco['annotations'] if ann.get('image_id') == image_id]
                for ann in current_anns:
                    seg = ann.get('segmentation', None)
                    if seg is None:
                        continue
                    color = (0, 255, 0) if int(ann.get('category_id', -1)) == 0 else (255, 0, 0)
                    # Polygon 표시 (RLE는 폴백)
                    if isinstance(seg, list) and len(seg) > 0:
                        for poly in seg:
                            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                            if pts.shape[0] >= 3:
                                cv2.polylines(vis_img, [pts.astype(np.int32)], True, color, 2)
                    elif isinstance(seg, dict) and 'counts' in seg:
                        try:
                            mask = self._mask_from_rle(seg.get('counts', []), seg.get('size', [h, w]))
                            if mask is not None:
                                ys, xs = np.where(mask > 0)
                                if len(xs) > 0:
                                    pts = np.column_stack([xs, ys]).astype(np.int32)
                                    cv2.polylines(vis_img, [pts], True, color, 1)
                        except Exception:
                            pass
                cv2.imwrite(os.path.join(debug_dir, base_name + "_coco_viz.jpg"), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

            messagebox.showinfo("성공", f"COCO Seg 내보내기 완료:\n{out_dir}")
            self._safe_refocus()
        except Exception as e:
            messagebox.showerror("오류", f"COCO Seg 내보내기 실패:\n{e}")
            self._safe_refocus()

    def export_csv(self):
        """CSV 내보내기 (삭제된 객체 제외)"""
        if not self.analysis_results:
            messagebox.showerror("오류", "분석 결과가 없습니다. 먼저 분석을 실행해주세요.")
            self._safe_refocus()
            return
        file_path = filedialog.asksaveasfilename(
            title="CSV 파일로 저장",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        self._safe_refocus()
        if not file_path:
            return
        try:
            # 삭제되지 않은 객체만 필터링
            filtered_objects = [
                obj for obj in self.analysis_results['objects']
                if obj.get('id', 0) not in self._deleted_objects
            ]

            if len(filtered_objects) == 0:
                messagebox.showwarning("경고", "내보낼 객체가 없습니다. (모두 삭제됨)")
                self._safe_refocus()
                return

            # pixels_per_cm2 재계산
            pixels_per_cm2_val, has_scale = self._get_export_scale_factor()
            if not has_scale or not pixels_per_cm2_val:
                pixels_per_cm2_val = self.analysis_results.get('pixels_per_cm2', 1.0) if self.analysis_results else 1.0
            pixels_per_cm2_val = float(pixels_per_cm2_val) if pixels_per_cm2_val else 1.0
            
            # 길이 변환 계수 (면적의 제곱근)
            pixels_per_cm = np.sqrt(pixels_per_cm2_val) if has_scale and pixels_per_cm2_val > 0 else None

            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header: Scale 있을 때만 cm 단위 추가
                if pixels_per_cm:
                    writer.writerow(['Leaf_ID', 'Area_pixels', 'Area_cm2', 'Length_pixels', 'Length_cm', 'Width_pixels', 'Width_cm', 'Perimeter_pixels', 'Perimeter_cm', 'Center_X', 'Center_Y', 'Status'])
                else:
                    writer.writerow(['Leaf_ID', 'Area_pixels', 'Length_pixels', 'Width_pixels', 'Perimeter_pixels', 'Center_X', 'Center_Y', 'Status'])

                # Leaf objects
                for obj in filtered_objects:
                    center_x, center_y = obj.get("center", (0, 0))
                    
                    if pixels_per_cm:
                        writer.writerow([
                            obj.get('id', 0),
                            obj.get('area', 0),
                            obj.get('area', 0) / pixels_per_cm2_val,
                            obj.get('length', 0),
                            obj.get('length', 0) / pixels_per_cm,
                            obj.get('width', 0),
                            obj.get('width', 0) / pixels_per_cm,
                            obj.get('perimeter', 0),
                            obj.get('perimeter', 0) / pixels_per_cm,
                            center_x,
                            center_y,
                            'Active'
                        ])
                    else:
                        writer.writerow([
                            obj.get('id', 0),
                            obj.get('area', 0),
                            obj.get('length', 0),
                            obj.get('width', 0),
                            obj.get('perimeter', 0),
                            center_x,
                            center_y,
                            'Active'
                        ])

                # Scale info
                if self._current_scale_labels is not None:
                    # Scale 라벨맵이 있는 경우
                    scale_labels = self._current_scale_labels
                    unique_scale_ids = np.unique(scale_labels)
                    active_scale_objects = 0

                    writer.writerow([])
                    writer.writerow(['=== Scale Information ==='])
                    writer.writerow(['Scale_ID', 'Area_pixels', 'Area_cm2', 'Center_X', 'Center_Y', 'Status'])

                    for scale_id in unique_scale_ids:
                        if scale_id > 0 and scale_id not in self._deleted_scale_objects:
                            scale_mask = (scale_labels == scale_id)
                            scale_area = np.sum(scale_mask)

                            if scale_area > 0:
                                # Scale 객체 중심 계산
                                y_coords, x_coords = np.where(scale_mask)
                                center_x = int(np.mean(x_coords))
                                center_y = int(np.mean(y_coords))

                                writer.writerow([
                                    f'S{scale_id}',
                                    scale_area,
                                    (scale_area / pixels_per_cm2_val) if pixels_per_cm2_val else "",
                                    center_x,
                                    center_y,
                                    'Active'
                                ])
                                active_scale_objects += 1

                    if active_scale_objects == 0:
                        writer.writerow(['No active scale objects'])
                elif 'scale_mask' in self.analysis_results:
                    # 기본 분석 경로: 라벨맵이 없어도 scale_mask 연결 성분으로 목록 생성
                    try:
                        writer.writerow([])
                        writer.writerow(['=== Scale Information ==='])
                        writer.writerow(['Scale_ID', 'Area_pixels', 'Area_cm2', 'Center_X', 'Center_Y', 'Status'])
                        sm = self.analysis_results.get('scale_mask')
                        sm = np.asarray(sm).astype(np.uint8)
                        if sm.size > 0 and int(np.sum(sm)) > 0:
                            num_labels, labels = cv2.connectedComponents(sm, connectivity=8)
                            active_scale_objects = 0
                            for sid in range(1, num_labels):
                                mask_sid = (labels == sid)
                                scale_area = int(np.sum(mask_sid))
                                if scale_area <= 0:
                                    continue
                                ys, xs = np.where(mask_sid)
                                cx = int(np.mean(xs)) if xs.size > 0 else 0
                                cy = int(np.mean(ys)) if ys.size > 0 else 0
                                writer.writerow([
                                    f'S{sid}',
                                    scale_area,
                                    (scale_area / pixels_per_cm2_val) if pixels_per_cm2_val else "",
                                    cx,
                                    cy,
                                    'Active'
                                ])
                                active_scale_objects += 1
                            if active_scale_objects == 0:
                                writer.writerow(['No active scale objects'])
                        else:
                            writer.writerow(['No active scale objects'])
                    except Exception:
                        pass

            total_exported = len(filtered_objects)
            total_deleted = len(self._deleted_objects) + len(self._deleted_scale_objects)

            # 팝업: Leaf/Scale 분리 요약
            if self._current_scale_labels is not None:
                active_scale = len([sid for sid in np.unique(self._current_scale_labels) if sid > 0 and sid not in self._deleted_scale_objects])
                total_scale = len([sid for sid in np.unique(self._current_scale_labels) if sid > 0])
            else:
                active_scale = int(np.sum(self.analysis_results.get('scale_mask', np.zeros((1,1),dtype=bool)))) > 0
                total_scale = 1 if active_scale else 0
            msg = (
                f"CSV 파일이 저장되었습니다:\n{file_path}\n\n"
                f"Leaf: 내보낸 {total_exported}개, 숨김 {len(self._deleted_objects)}개\n"
                f"Scale: 내보낸 {active_scale}개, 숨김 {len(self._deleted_scale_objects)}개 (전체 {total_scale}개)"
            )
            messagebox.showinfo("성공", msg)
            self._safe_refocus()

        except Exception as e:
            messagebox.showerror("오류", f"CSV 저장 중 오류가 발생했습니다:\n{e}")
            self._safe_refocus()

    def export_json(self):
        """JSON 내보내기 (삭제된 객체 제외)"""
        if not self.analysis_results:
            messagebox.showerror("오류", "분석 결과가 없습니다. 먼저 분석을 실행해주세요.")
            self._safe_refocus()
            return

        # 삭제되지 않은 객체만 필터링
        filtered_objects = [
            obj for obj in self.analysis_results['objects']
            if obj.get('id', 0) not in self._deleted_objects
        ]

        if len(filtered_objects) == 0:
            messagebox.showwarning("경고", "내보낼 객체가 없습니다. (모두 삭제됨)")
            self._safe_refocus()
            return

        file_path = filedialog.asksaveasfilename(
            title="JSON 파일로 저장",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        self._safe_refocus()

        if not file_path:
            return

        try:
            # JSON 직렬화를 위해 numpy 배열 등을 변환 (삭제된 객체 제외)
            total_filtered_area = sum(obj.get('area', 0) for obj in filtered_objects)

            # Scale 객체 정보 수집
            scale_objects = []
            scale_total_area = 0

            if self._current_scale_labels is not None and 'scale_mask' in self.analysis_results:
                scale_labels = self._current_scale_labels
                unique_scale_ids = np.unique(scale_labels)

                for scale_id in unique_scale_ids:
                    if scale_id > 0 and scale_id not in self._deleted_scale_objects:
                        scale_mask = (scale_labels == scale_id)
                        scale_area = np.sum(scale_mask)

                        if scale_area > 0:
                            # Scale 객체 중심 계산
                            y_coords, x_coords = np.where(scale_mask)
                            center_x = float(np.mean(x_coords))
                            center_y = float(np.mean(y_coords))

                            # area_cm2는 내보내기 시점 스케일 팩터 적용
                            scale_obj = {
                                'id': f'S{scale_id}',
                                'area_pixels': int(scale_area),
                                'area_cm2': None,  # summary에서 일괄 처리됨; 개별도 원하면 아래 주석 해제
                                'center': [center_x, center_y],
                                'status': 'active'
                            }
                            scale_objects.append(scale_obj)
                            scale_total_area += scale_area

            # 내보내기 시점의 pixels_per_cm2 재계산 (없으면 None)
            pixels_per_cm2_export, has_scale = self._get_export_scale_factor()

            # linear conversion factor
            pixels_per_cm = float(np.sqrt(pixels_per_cm2_export)) if (has_scale and pixels_per_cm2_export and pixels_per_cm2_export > 0) else None

            export_data = {
                "summary": {
                    "total_leaf_objects": len(filtered_objects),
                    "total_leaf_area_pixels": total_filtered_area,
                    "total_leaf_area_cm2": (total_filtered_area / pixels_per_cm2_export) if (has_scale and pixels_per_cm2_export) else None,
                    "total_scale_objects": len(scale_objects),
                    "total_scale_area_pixels": scale_total_area,
                    "total_scale_area_cm2": (scale_total_area / pixels_per_cm2_export) if (has_scale and pixels_per_cm2_export) else None,
                    "pixels_per_cm2": pixels_per_cm2_export if (has_scale and pixels_per_cm2_export) else None,
                    "pixels_per_cm": pixels_per_cm if pixels_per_cm else None,
                    "deleted_leaf_objects_count": len(self._deleted_objects),
                    "deleted_leaf_object_ids": sorted(list(self._deleted_objects)),
                    "deleted_scale_objects_count": len(self._deleted_scale_objects),
                    "deleted_scale_object_ids": sorted(list(self._deleted_scale_objects))
                },
                "leaf_objects": [],
                "scale_objects": scale_objects
            }

            for obj in filtered_objects:
                obj_data = obj.copy()

                # numpy array conversions
                if 'center' in obj_data:
                    obj_data['center'] = [float(obj_data['center'][0]), float(obj_data['center'][1])]
                if 'bounding_box' in obj_data:
                    obj_data['bounding_box'] = obj_data['bounding_box'].tolist()
                if 'contour' in obj_data:
                    del obj_data['contour']

                # rename pixel fields for clarity
                if 'area' in obj_data:
                    obj_data['area_pixels'] = obj_data.pop('area')
                if 'length' in obj_data:
                    obj_data['length_pixels'] = obj_data.pop('length')
                if 'width' in obj_data:
                    obj_data['width_pixels'] = obj_data.pop('width')
                if 'perimeter' in obj_data:
                    obj_data['perimeter_pixels'] = obj_data.pop('perimeter')

                # add cm fields
                obj_data['area_cm2'] = (float(obj_data.get('area_pixels', 0.0)) / pixels_per_cm2_export) if (has_scale and pixels_per_cm2_export) else None
                obj_data['length_cm'] = (float(obj_data.get('length_pixels', 0.0)) / pixels_per_cm) if pixels_per_cm else None
                obj_data['width_cm'] = (float(obj_data.get('width_pixels', 0.0)) / pixels_per_cm) if pixels_per_cm else None
                obj_data['perimeter_cm'] = (float(obj_data.get('perimeter_pixels', 0.0)) / pixels_per_cm) if pixels_per_cm else None

                export_data["leaf_objects"].append(obj_data)

            # JSON 직렬화 안전 변환 적용 (numpy 타입 대응)
            export_data_safe = self._json_safe(export_data)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data_safe, f, indent=2, ensure_ascii=False)

            total_exported_leaf = len(filtered_objects)
            total_exported_scale = len(scale_objects)
            total_deleted = len(self._deleted_objects) + len(self._deleted_scale_objects)

            # 팝업: Leaf/Scale 분리 요약
            if self._current_scale_labels is not None:
                active_scale = len([sid for sid in np.unique(self._current_scale_labels) if sid > 0 and sid not in self._deleted_scale_objects])
                total_scale = len([sid for sid in np.unique(self._current_scale_labels) if sid > 0])
            else:
                active_scale = int(np.sum(self.analysis_results.get('scale_mask', np.zeros((1,1),dtype=bool)))) > 0
                total_scale = 1 if active_scale else 0
            msg = (
                f"JSON 파일이 저장되었습니다:\n{file_path}\n\n"
                f"Leaf: 내보낸 {total_exported_leaf}개, 숨김 {len(self._deleted_objects)}개\n"
                f"Scale: 내보낸 {active_scale}개, 숨김 {len(self._deleted_scale_objects)}개 (전체 {total_scale}개)"
            )
            messagebox.showinfo("성공", msg)

        except Exception as e:
            messagebox.showerror("오류", f"JSON 저장 중 오류가 발생했습니다:\n{e}")
