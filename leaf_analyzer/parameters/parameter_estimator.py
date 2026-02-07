#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Estimator for Advanced Leaf Analyzer
파라미터 자동 추정 및 조정

개선된 알고리즘:
- elaMac2024.py의 반복적 적응 알고리즘 기반
- NumPy 벡터화로 속도 향상
- 병렬 파라미터 완화
- 적응형 목표 픽셀 수
- 스케일 자동 검출 포함
- 캘리브레이션 파일 의존성 제거
"""

import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox

try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    ctk = tk
    CTK_AVAILABLE = False


class ParameterEstimator:
    """파라미터 추정 및 관리 믹스인 클래스"""

    def _estimate_params_from_image(self) -> dict:
        """이미지 기반 자동 파라미터 추정 (개선된 반복적 적응 알고리즘)
        
        elaMac2024.py의 auto_Settings() 알고리즘을 기반으로:
        1. 엄격한 초기값에서 시작
        2. 목표 픽셀 수에 도달할 때까지 파라미터 완화
        3. 검출된 픽셀의 평균 특성으로 최종 파라미터 계산
        4. 스케일(빨간색) 파라미터도 자동 추정
        
        개선 사항:
        - NumPy 벡터화로 속도 향상
        - 병렬 파라미터 완화 (ratG, ratGb, minG 동시에)
        - 적응형 목표 픽셀 수
        - 캘리브레이션 파일 없이 직접 계산
        
        Returns:
            dict: 추정된 파라미터 {"minG", "ratG", "ratGb", "minR", "ratR"}
        """
        if self.original_image is None:
            return {}
        
        img = self.working_image if hasattr(self, 'working_image') and self.working_image is not None else self.original_image
        h, w = img.shape[:2]
        total_pixels = h * w
        
        # RGB 채널 분리 (float32로 변환하여 정밀도 확보)
        r = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        b = img[:, :, 2].astype(np.float32)
        
        # ========== 잎(녹색) 파라미터 추정 ==========
        leaf_params = self._estimate_leaf_params_iterative(r, g, b, total_pixels)
        
        # ========== 스케일 파라미터 추정 (색상에 따라 다름) ==========
        scale_color = getattr(self, 'scale_color_var', None)
        scale_mode = scale_color.get() if scale_color else "red"
        
        if scale_mode == "blue":
            scale_params = self._estimate_blue_scale_params_iterative(r, g, b, total_pixels)
        else:
            scale_params = self._estimate_scale_params_iterative(r, g, b, total_pixels)
        
        # 결과 병합
        estimated = {**leaf_params, **scale_params}
        
        print(f"[자동 파라미터] 잎: minG={estimated['minG']}, ratG={estimated['ratG']:.2f}, ratGb={estimated['ratGb']:.2f}")
        print(f"[자동 파라미터] 스케일 ({scale_mode}): minR={estimated['minR']}, ratR={estimated['ratR']:.2f}")
        
        return estimated
    
    def _estimate_leaf_params_iterative(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, 
                                         total_pixels: int) -> dict:
        """반복적 적응으로 잎(녹색) 파라미터 추정
        
        elaMac2024.py의 알고리즘을 기반으로 하되, 과검출을 방지하기 위해
        더 보수적인 파라미터를 사용합니다.
        
        Args:
            r, g, b: RGB 채널 (float32)
            total_pixels: 전체 픽셀 수
            
        Returns:
            dict: {"minG": int, "ratG": float, "ratGb": float}
        """
        # 적응형 목표: 이미지의 0.1% ~ 1.0% 범위에서 동적 설정
        # 기본 목표는 0.25% (elaMac2024.py와 동일)
        target_min = int(total_pixels * 0.001)   # 최소 0.1%
        target_default = int(total_pixels * 0.0025)  # 기본 0.25%
        
        # 초기값 (엄격한 조건에서 시작 - elaMac2024.py와 동일)
        ratG = 2.0
        ratGb = 1.8
        minG = 75.0
        
        # 완화 계수 (elaMac2024.py와 동일)
        decay_ratio = 0.94
        decay_minG = 0.90
        
        max_iterations = 15
        leaf_mask = None
        cnt = 0
        
        for loop in range(max_iterations):
            # NumPy 벡터화 조건: G > R*ratG AND G > B*ratGb AND G > minG
            leaf_mask = (r * ratG < g) & (b * ratGb < g) & (g > minG)
            cnt = np.sum(leaf_mask)
            
            if cnt >= target_default:
                break
            
            # elaMac2024.py 스타일 완화
            if loop < 12:
                ratG *= decay_ratio
                ratGb *= decay_ratio
            else:
                minG *= decay_minG
            
            # 최소값 제한 (관대하게 - 다양한 잎 색상 포함)
            # ratG < 1.0 허용: G가 R과 거의 같아도 잎일 수 있음
            ratG = max(ratG, 0.9)
            ratGb = max(ratGb, 0.85)
            minG = max(minG, 10.0)
        
        print(f"[잎 검출] {loop+1}회 반복 후 {cnt}개 픽셀 검출 (목표: {target_default})")
        
        # 검출된 픽셀이 너무 적으면 기본값 반환
        if cnt < target_min:
            print("[잎 검출] 픽셀 부족, 기본값 사용")
            return {"minG": 25, "ratG": 1.06, "ratGb": 1.08}
        
        # 검출된 픽셀의 특성으로 최종 파라미터 계산
        leaf_g = g[leaf_mask]
        leaf_r = r[leaf_mask]
        leaf_b = b[leaf_mask]
        
        # 0 나눗셈 방지
        leaf_r_safe = np.where(leaf_r < 1, leaf_g, leaf_r)
        leaf_b_safe = np.where(leaf_b < 1, leaf_g, leaf_b)
        
        # 평균 계산 (elaMac2024.py와 동일)
        g_avg = np.mean(leaf_g)
        gr_avg = np.mean(leaf_g / leaf_r_safe)  # G/R 비율 평균
        gb_avg = np.mean(leaf_g / leaf_b_safe)  # G/B 비율 평균
        
        # ========== elaMac2024.py 기본 보정 계수 적용 (완화 버전) ==========
        # calib.csv가 없을 때 사용되는 Arabidopsis 기본값에 여유 마진 적용
        # 다양한 식물 종에 대응하기 위해 비율 조건을 완화
        
        # minG = mg * gavg + bg = 1.223 * gavg - 111
        # 최소값 20
        final_minG = int(np.clip(1.223 * g_avg - 111, 20, 180))
        
        # ratG, ratGb: 보정 공식 결과에 0.75배 감쇄 적용 (더 관대하게)
        # 잎 내부의 잎맥, 밝은 부분, 다양한 녹색 포함
        raw_ratG = 0.360 * gr_avg + 0.589
        raw_ratGb = 0.334 * gb_avg + 0.534
        
        # 0.75배 감쇄 + 낮은 최소값
        final_ratG = float(np.clip(raw_ratG * 0.75, 0.85, 1.5))
        final_ratGb = float(np.clip(raw_ratGb * 0.75, 0.8, 1.4))
        
        return {
            "minG": final_minG,
            "ratG": round(final_ratG, 2),
            "ratGb": round(final_ratGb, 2)
        }
    
    def _estimate_scale_params_iterative(self, r: np.ndarray, g: np.ndarray, b: np.ndarray,
                                          total_pixels: int) -> dict:
        """반복적 적응으로 스케일(빨간색) 파라미터 추정
        
        스케일은 일반적으로 이미지의 매우 작은 부분만 차지하므로
        보수적인 파라미터를 사용하여 과검출을 방지합니다.
        
        Args:
            r, g, b: RGB 채널 (float32)
            total_pixels: 전체 픽셀 수
            
        Returns:
            dict: {"minR": int, "ratR": float}
        """
        # 스케일용 목표: 매우 작게 설정 (0.005% ~ 0.05%)
        # 스케일은 보통 이미지의 아주 작은 부분만 차지
        target_min = max(100, int(total_pixels * 0.00005))   # 최소 0.005% 또는 100픽셀
        target_default = max(500, int(total_pixels * 0.0005))  # 기본 0.05% 또는 500픽셀
        
        # 초기값 (엄격한 조건 - elaMac2024.py와 동일)
        ratR = 2.0
        minR = 150.0
        
        # 완화 계수 (더 보수적으로)
        decay_ratio = 0.96  # 0.94 → 0.96 (더 천천히 완화)
        decay_minR = 0.98   # 0.97 → 0.98
        
        max_iterations = 10  # 12 → 10 (조기 종료)
        scale_mask = None
        cnt = 0
        
        for loop in range(max_iterations):
            # NumPy 벡터화 조건: R > G*ratR AND R > B*ratR AND R > minR
            scale_mask = (g * ratR < r) & (b * ratR < r) & (r > minR)
            cnt = np.sum(scale_mask)
            
            if cnt >= target_default:
                break
            
            # 완화 (보수적)
            ratR *= decay_ratio
            minR *= decay_minR
            
            # 최소값 제한 (더 엄격하게)
            # ratR >= 1.2: R이 G/B보다 최소 20% 이상 커야 함
            ratR = max(ratR, 1.2)
            minR = max(minR, 150.0)
        
        print(f"[스케일 검출] {loop+1}회 반복 후 {cnt}개 픽셀 검출 (목표: {target_default})")
        
        # 스케일 미검출 또는 과검출 시 기본값
        if cnt < target_min:
            print("[스케일 검출] 스케일 미검출, 기본값 사용 (minR=255, ratR=2.0)")
            return {"minR": 255, "ratR": 2.0}
        
        # 과검출 방지: 이미지의 5% 이상이면 스케일이 아님
        if cnt > total_pixels * 0.05:
            print(f"[스케일 검출] 과검출 감지 ({cnt}픽셀 = {100*cnt/total_pixels:.2f}%), 기본값 사용")
            return {"minR": 255, "ratR": 2.0}
        
        # 검출된 픽셀의 평균 특성으로 최종 파라미터 계산
        scale_r = r[scale_mask]
        scale_g = g[scale_mask]
        scale_b = b[scale_mask]
        
        # 0 나눗셈 방지
        scale_g_safe = np.where(scale_g < 1, scale_r, scale_g)
        scale_b_safe = np.where(scale_b < 1, scale_r, scale_b)
        
        # 평균 계산 (elaMac2024.py와 동일)
        r_avg = np.mean(scale_r)
        rg_avg = np.mean(scale_r / scale_g_safe)  # R/G 비율 평균
        rb_avg = np.mean(scale_r / scale_b_safe)  # R/B 비율 평균
        
        # ========== elaMac2024.py 기본 보정 계수 적용 ==========
        # mmr=1.412, bmr=-140.6, mmg=0.134, bmg=0.782
        
        # minR = mmr * ravg + bmr = 1.412 * ravg - 140.6
        final_minR = int(np.clip(1.412 * r_avg - 140.6, 100, 255))
        
        # ratR = mmg * rgavg + bmg = 0.134 * ((rg_avg+rb_avg)/2) + 0.782
        rr_avg = (rg_avg + rb_avg) / 2.0
        final_ratR = float(np.clip(0.134 * rr_avg + 0.782, 1.0, 2.0))
        
        return {
            "minR": final_minR,
            "ratR": round(final_ratR, 2)
        }
    
    def _estimate_blue_scale_params_iterative(self, r: np.ndarray, g: np.ndarray, b: np.ndarray,
                                               total_pixels: int) -> dict:
        """반복적 적응으로 스케일(파란색) 파라미터 추정
        
        파란색 스케일 검출: B > minB AND B > R*ratB AND B > G*ratB AND R < max_r AND G < max_g
        진한 파란색(dark blue)도 감지할 수 있도록 조건 완화
        
        Args:
            r, g, b: RGB 채널 (float32)
            total_pixels: 전체 픽셀 수
            
        Returns:
            dict: {"minB": int, "ratB": float, "blue_max_r": int, "blue_max_g": int}
        """
        # 스케일용 목표: 매우 작게 설정 (0.005% ~ 0.1%)
        target_min = max(100, int(total_pixels * 0.00005))
        target_default = max(500, int(total_pixels * 0.001))
        
        # 초기값 (진한 파란색도 감지할 수 있도록 완화된 시작점)
        ratB = 1.8
        minB = 60.0
        max_r = 180.0
        max_g = 180.0
        
        # 완화 계수
        decay_ratio = 0.95
        decay_minB = 0.92
        
        max_iterations = 15
        scale_mask = None
        cnt = 0
        
        for loop in range(max_iterations):
            # NumPy 벡터화 조건: B가 지배적이고 R, G가 낮은 영역
            scale_mask = (
                (b > minB) & 
                (b > r * ratB) & 
                (b > g * ratB) & 
                (r < max_r) & 
                (g < max_g)
            )
            cnt = np.sum(scale_mask)
            
            if cnt >= target_default:
                break
            
            # 완화
            ratB *= decay_ratio
            minB *= decay_minB
            
            # 최소값 제한 (진한 파란색도 감지)
            ratB = max(ratB, 1.1)
            minB = max(minB, 40.0)
        
        print(f"[파란 스케일 검출] {loop+1}회 반복 후 {cnt}개 픽셀 검출 (목표: {target_default})")
        
        # 스케일 미검출 시 기본값
        if cnt < target_min:
            print("[파란 스케일 검출] 스케일 미검출, 기본값 사용")
            return {
                "minB": 80,
                "ratB": 1.3,
                "blue_max_r": 150,
                "blue_max_g": 150,
                "minR": 180,  # 빨간 스케일 기본값도 설정
                "ratR": 1.5
            }
        
        # 과검출 방지: 이미지의 5% 이상이면 스케일이 아님
        if cnt > total_pixels * 0.05:
            print(f"[파란 스케일 검출] 과검출 감지 ({cnt}픽셀 = {100*cnt/total_pixels:.2f}%), 기본값 사용")
            return {
                "minB": 100,
                "ratB": 1.5,
                "blue_max_r": 120,
                "blue_max_g": 120,
                "minR": 180,
                "ratR": 1.5
            }
        
        # 검출된 픽셀의 평균 특성으로 최종 파라미터 계산
        scale_r = r[scale_mask]
        scale_g = g[scale_mask]
        scale_b = b[scale_mask]
        
        # 평균 계산 (0 나눗셈 방지)
        b_avg = np.mean(scale_b)
        r_avg = np.mean(scale_r)
        g_avg = np.mean(scale_g)
        
        scale_r_safe = np.where(scale_r < 1, 1, scale_r)
        scale_g_safe = np.where(scale_g < 1, 1, scale_g)
        
        br_avg = np.mean(scale_b / scale_r_safe)  # B/R 비율 평균
        bg_avg = np.mean(scale_b / scale_g_safe)  # B/G 비율 평균
        
        # 최종 파라미터 계산
        # minB: 검출된 B값의 하위 10% 기준 (여유 있게)
        final_minB = int(np.clip(np.percentile(scale_b, 10) * 0.8, 40, 200))
        
        # ratB: 평균 비율의 80% (여유 있게)
        bb_avg = min(br_avg, bg_avg)
        final_ratB = float(np.clip(bb_avg * 0.8, 1.1, 2.0))
        
        # max_r, max_g: 검출된 R, G값의 상위 90% + 여유
        final_max_r = int(np.clip(np.percentile(scale_r, 90) + 30, 100, 200))
        final_max_g = int(np.clip(np.percentile(scale_g, 90) + 30, 100, 200))
        
        print(f"   → 파란 스케일 최종 파라미터: minB={final_minB}, ratB={final_ratB:.2f}, max_r={final_max_r}, max_g={final_max_g}")
        print(f"   → 검출 픽셀 평균: B={b_avg:.1f}, R={r_avg:.1f}, G={g_avg:.1f}")
        
        return {
            "minB": final_minB,
            "ratB": round(final_ratB, 2),
            "blue_max_r": final_max_r,
            "blue_max_g": final_max_g,
            "minR": 180,  # 빨간 스케일 기본값
            "ratR": 1.5
        }
    
    def _estimate_params_from_seeds(self) -> dict:
        """시드 주변 픽셀 기반 파라미터 추정
        
        시드가 있으면 시드 주변 영역을 분석하여 더 정밀하게 파라미터 추정.
        시드 부족 시 이미지 전체 분석으로 폴백.
        
        Returns:
            dict: 추정된 파라미터 {"minG", "ratG", "ratGb", "minR", "ratR"}
        """
        if self.original_image is None:
            return {}
        
        leaf_seeds = self.seed_manager.seeds.get("leaf", [])
        scale_seeds = self.seed_manager.seeds.get("scale", [])
        
        # 시드 부족 시 이미지 전체 분석으로 폴백
        if len(leaf_seeds) < 3:
            print("[시드 기반] 잎 시드 부족, 이미지 전체 분석으로 폴백")
            return self._estimate_params_from_image()
        
        img = self.working_image if hasattr(self, 'working_image') and self.working_image is not None else self.original_image
        h, w = img.shape[:2]
        
        # ========== 잎 시드 기반 파라미터 ==========
        leaf_params = self._estimate_params_from_seed_regions(
            img, leaf_seeds, h, w, seed_type="leaf"
        )
        
        # ========== 스케일 시드 기반 파라미터 ==========
        if len(scale_seeds) >= 1:
            scale_params = self._estimate_params_from_seed_regions(
                img, scale_seeds, h, w, seed_type="scale"
            )
        else:
            # 스케일 시드 없으면 이미지 전체에서 추정
            r = img[:, :, 0].astype(np.float32)
            g = img[:, :, 1].astype(np.float32)
            b = img[:, :, 2].astype(np.float32)
            scale_params = self._estimate_scale_params_iterative(r, g, b, h * w)
        
        # 결과 병합
        estimated = {**leaf_params, **scale_params}
        
        print(f"[시드 기반] 잎: minG={estimated['minG']}, ratG={estimated['ratG']:.2f}, ratGb={estimated['ratGb']:.2f}")
        print(f"[시드 기반] 스케일: minR={estimated['minR']}, ratR={estimated['ratR']:.2f}")
        
        return estimated
    
    def _estimate_params_from_seed_regions(self, img: np.ndarray, seeds: list, 
                                            h: int, w: int, seed_type: str) -> dict:
        """시드 주변 영역에서 파라미터 추정
        
        Args:
            img: 이미지 배열
            seeds: 시드 좌표 리스트
            h, w: 이미지 높이, 너비
            seed_type: "leaf" 또는 "scale"
            
        Returns:
            dict: 추정된 파라미터
        """
        sample_radius = 25  # 시드 주변 25픽셀 반경
        
        # 시드 주변 픽셀 수집
        samples = []
        for (sx, sy) in seeds:
            sx, sy = int(sx), int(sy)
            if not (0 <= sx < w and 0 <= sy < h):
                continue
            
            y1 = max(0, sy - sample_radius)
            y2 = min(h, sy + sample_radius)
            x1 = max(0, sx - sample_radius)
            x2 = min(w, sx + sample_radius)
            
            roi = img[y1:y2, x1:x2]
            roi_pixels = roi.reshape(-1, 3).astype(np.float32)
            samples.extend(roi_pixels)
        
        if len(samples) < 100:
            # 샘플 부족 시 기본값 반환
            if seed_type == "leaf":
                return {"minG": 25, "ratG": 1.06, "ratGb": 1.08}
            else:
                return {"minR": 225, "ratR": 1.94}
        
        samples = np.array(samples)
        r_samples = samples[:, 0]
        g_samples = samples[:, 1]
        b_samples = samples[:, 2]
        
        if seed_type == "leaf":
            # 잎 파라미터: G 채널 기준
            # 이상치 제거 (IQR 기반)
            g_q1, g_q3 = np.percentile(g_samples, [25, 75])
            g_iqr = g_q3 - g_q1
            valid_mask = (g_samples >= g_q1 - 1.5 * g_iqr) & (g_samples <= g_q3 + 1.5 * g_iqr)
            
            if np.sum(valid_mask) > 50:
                r_samples = r_samples[valid_mask]
                g_samples = g_samples[valid_mask]
                b_samples = b_samples[valid_mask]
            
            # 파라미터 계산
            r_safe = np.where(r_samples < 1, g_samples, r_samples)
            b_safe = np.where(b_samples < 1, g_samples, b_samples)
            
            g_avg = np.mean(g_samples)
            gr_avg = np.mean(g_samples / r_safe)
            gb_avg = np.mean(g_samples / b_safe)
            
            # 시드 영역 기반이므로 약간 더 관대하게 설정
            final_minG = int(np.clip(0.6 * g_avg, 10, 180))
            final_ratG = float(np.clip(0.85 * gr_avg, 1.0, 1.8))
            final_ratGb = float(np.clip(0.85 * gb_avg, 0.9, 2.0))
            
            return {
                "minG": final_minG,
                "ratG": round(final_ratG, 2),
                "ratGb": round(final_ratGb, 2)
            }
        else:
            # 스케일 파라미터: R 채널 기준
            r_q1, r_q3 = np.percentile(r_samples, [25, 75])
            r_iqr = r_q3 - r_q1
            valid_mask = (r_samples >= r_q1 - 1.5 * r_iqr) & (r_samples <= r_q3 + 1.5 * r_iqr)
            
            if np.sum(valid_mask) > 50:
                r_samples = r_samples[valid_mask]
                g_samples = g_samples[valid_mask]
                b_samples = b_samples[valid_mask]
            
            g_safe = np.where(g_samples < 1, r_samples, g_samples)
            b_safe = np.where(b_samples < 1, r_samples, b_samples)
            
            r_avg = np.mean(r_samples)
            rg_avg = np.mean(r_samples / g_safe)
            rb_avg = np.mean(r_samples / b_safe)
            
            final_minR = int(np.clip(0.8 * r_avg, 100, 250))
            rr_avg = (rg_avg + rb_avg) / 2.0
            final_ratR = float(np.clip(0.9 * rr_avg, 1.01, 2.0))
            
            return {
                "minR": final_minR,
                "ratR": round(final_ratR, 2)
            }

    def reset_auto_params(self):
        """자동 파라미터 추정 모드로 리셋"""
        self._user_manually_adjusted_params = False
        
        # 기본값으로 리셋 (간소화된 파라미터)
        self.easy_params["minG"] = 25
        self.easy_params["ratG"] = 1.06
        self.easy_params["ratGb"] = 1.08
        # 빨간 스케일
        self.easy_params["minR"] = 180
        self.easy_params["ratR"] = 1.5
        # 파란 스케일
        self.easy_params["minB"] = 80
        self.easy_params["ratB"] = 1.3
        self.easy_params["blue_max_r"] = 150
        self.easy_params["blue_max_g"] = 150
        # 필터링
        self.easy_params["min_component"] = 500
        self.easy_params["min_green_diff"] = 10
        self.easy_params["dark_ratio_mult"] = 1.25
        self.easy_params["white_ratio_mult"] = 0.9
        
        # UI 레이블 업데이트
        if hasattr(self, 'easy_params_label'):
            self.easy_params_label.configure(
                text=f"G>{self.easy_params['minG']}, G/R>{self.easy_params['ratG']:.2f}, G/B>{self.easy_params['ratGb']:.2f}"
            )
        
        messagebox.showinfo(
            "자동 파라미터 리셋", 
            "자동 파라미터 추정 모드가 활성화되었습니다.\n\n"
            "모든 파라미터가 기본값으로 복구되었습니다.\n\n"
            "다음 기본 분석 시:\n"
            "• 시드 3개 이상: 시드 기반 정밀 추정\n"
            "• 시드 없음: 이미지 전체 반복적 적응 추정"
        )
        self._safe_refocus()

    def adjust_easy_params(self):
        """기본 분석 파라미터 조정 다이얼로그"""
        if not CTK_AVAILABLE:
            messagebox.showinfo("정보", "파라미터 조정은 CustomTkinter가 필요합니다.")
            self._safe_refocus()
            return
            
        # 파라미터 설정 윈도우
        param_window = ctk.CTkToplevel(self.root)
        param_window.title("기본 분석 파라미터")
        param_window.geometry("420x700")
        param_window.grid_rowconfigure(0, weight=1)
        param_window.grid_columnconfigure(0, weight=1)
        
        content = ctk.CTkScrollableFrame(param_window)
        content.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        
        # 파라미터 슬라이더들
        ctk.CTkLabel(content, text="잎 검출 파라미터", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 최소 녹색값 (minG)
        minG_var = tk.IntVar(value=self.easy_params["minG"])
        ctk.CTkLabel(content, text="최소 녹색 RGB 값:").pack()
        minG_slider = ctk.CTkSlider(content, from_=0, to=255, variable=minG_var)
        minG_slider.pack(pady=5)
        minG_label = ctk.CTkLabel(content, text=str(self.easy_params["minG"]))
        minG_label.pack()
        
        def update_minG_label(value):
            try:
                v = int(float(value))
            except Exception:
                v = int(minG_var.get())
            minG_label.configure(text=str(v))
            self.easy_params["minG"] = v
            if hasattr(self, 'easy_params_label'):
                self.easy_params_label.configure(
                    text=f"G>{self.easy_params['minG']}, G/R>{self.easy_params['ratG']:.2f}, G/B>{self.easy_params['ratGb']:.2f}"
                )
        minG_slider.configure(command=update_minG_label)
        
        # G/R 비율 (ratG)
        ratG_var = tk.DoubleVar(value=self.easy_params["ratG"])
        ctk.CTkLabel(content, text="G/R 비율:").pack()
        ratG_slider = ctk.CTkSlider(content, from_=0.9, to=2.0, variable=ratG_var)
        ratG_slider.pack(pady=5)
        ratG_label = ctk.CTkLabel(content, text=f"{self.easy_params['ratG']:.2f}")
        ratG_label.pack()
        
        def update_ratG_label(value):
            try:
                v = float(value)
            except Exception:
                v = float(ratG_var.get())
            ratG_label.configure(text=f"{v:.2f}")
            self.easy_params["ratG"] = v
            if hasattr(self, 'easy_params_label'):
                self.easy_params_label.configure(
                    text=f"G>{self.easy_params['minG']}, G/R>{self.easy_params['ratG']:.2f}, G/B>{self.easy_params['ratGb']:.2f}"
                )
        ratG_slider.configure(command=update_ratG_label)
        
        # G/B 비율 (ratGb)
        ratGb_var = tk.DoubleVar(value=self.easy_params["ratGb"])
        ctk.CTkLabel(content, text="G/B 비율:").pack()
        ratGb_slider = ctk.CTkSlider(content, from_=0.8, to=2.0, variable=ratGb_var)
        ratGb_slider.pack(pady=5)
        ratGb_label = ctk.CTkLabel(content, text=f"{self.easy_params['ratGb']:.2f}")
        ratGb_label.pack()
        
        def update_ratGb_label(value):
            try:
                v = float(value)
            except Exception:
                v = float(ratGb_var.get())
            ratGb_label.configure(text=f"{v:.2f}")
            self.easy_params["ratGb"] = v
            if hasattr(self, 'easy_params_label'):
                self.easy_params_label.configure(
                    text=f"G>{self.easy_params['minG']}, G/R>{self.easy_params['ratG']:.2f}, G/B>{self.easy_params['ratGb']:.2f}"
                )
        ratGb_slider.configure(command=update_ratGb_label)
        
        ctk.CTkLabel(content, text="\n스케일 검출 파라미터", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 최소 빨간색값 (minR)
        minR_var = tk.IntVar(value=self.easy_params["minR"])
        ctk.CTkLabel(content, text="최소 빨간색 RGB 값:").pack()
        minR_slider = ctk.CTkSlider(content, from_=0, to=255, variable=minR_var)
        minR_slider.pack(pady=5)
        minR_label = ctk.CTkLabel(content, text=str(self.easy_params["minR"]))
        minR_label.pack()
        
        def update_minR_label(value):
            try:
                v = int(float(value))
            except Exception:
                v = int(minR_var.get())
            minR_label.configure(text=str(v))
            self.easy_params["minR"] = v
        minR_slider.configure(command=update_minR_label)
        
        # R/G, R/B 비율 (ratR)
        ratR_var = tk.DoubleVar(value=self.easy_params["ratR"])
        ctk.CTkLabel(content, text="R/G, R/B 비율:").pack()
        ratR_slider = ctk.CTkSlider(content, from_=1.0, to=2.0, variable=ratR_var)
        ratR_slider.pack(pady=5)
        ratR_label = ctk.CTkLabel(content, text=f"{self.easy_params['ratR']:.2f}")
        ratR_label.pack()
        
        def update_ratR_label(value):
            try:
                v = float(value)
            except Exception:
                v = float(ratR_var.get())
            ratR_label.configure(text=f"{v:.2f}")
            self.easy_params["ratR"] = v
        ratR_slider.configure(command=update_ratR_label)
        
        ctk.CTkLabel(content, text="\n배경색별 비율 계수", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 검은 배경 비율 계수
        dark_mult_var = tk.DoubleVar(value=self.easy_params.get("dark_ratio_mult", 1.25))
        ctk.CTkLabel(content, text="검은 배경 계수 (높을수록 엄격):").pack()
        dark_mult_slider = ctk.CTkSlider(content, from_=0.8, to=2.0, variable=dark_mult_var)
        dark_mult_slider.pack(pady=5)
        dark_mult_label = ctk.CTkLabel(content, text=f"{self.easy_params.get('dark_ratio_mult', 1.25):.2f}")
        dark_mult_label.pack()
        
        def update_dark_mult_label(value):
            try:
                v = float(value)
            except Exception:
                v = float(dark_mult_var.get())
            dark_mult_label.configure(text=f"{v:.2f}")
            self.easy_params["dark_ratio_mult"] = v
        dark_mult_slider.configure(command=update_dark_mult_label)
        
        # 흰색 배경 비율 계수
        white_mult_var = tk.DoubleVar(value=self.easy_params.get("white_ratio_mult", 0.9))
        ctk.CTkLabel(content, text="흰색 배경 계수 (낮을수록 관대):").pack()
        white_mult_slider = ctk.CTkSlider(content, from_=0.5, to=1.5, variable=white_mult_var)
        white_mult_slider.pack(pady=5)
        white_mult_label = ctk.CTkLabel(content, text=f"{self.easy_params.get('white_ratio_mult', 0.9):.2f}")
        white_mult_label.pack()
        
        def update_white_mult_label(value):
            try:
                v = float(value)
            except Exception:
                v = float(white_mult_var.get())
            white_mult_label.configure(text=f"{v:.2f}")
            self.easy_params["white_ratio_mult"] = v
        white_mult_slider.configure(command=update_white_mult_label)

        ctk.CTkLabel(content, text="\n후처리 파라미터", font=("Arial", 14, "bold")).pack(pady=10)
        morph_kernel_value = int(self.settings.get("morphology_kernel_size", self.manual_settings.get("morph_kernel", 5)))
        morph_kernel_var = tk.IntVar(value=morph_kernel_value)
        ctk.CTkLabel(content, text="모폴로지 커널 크기:").pack()
        morph_kernel_slider = ctk.CTkSlider(content, from_=3, to=15, variable=morph_kernel_var)
        morph_kernel_slider.pack(pady=5)
        morph_kernel_label = ctk.CTkLabel(content, text=str(morph_kernel_value))
        morph_kernel_label.pack()

        def update_morph_kernel_label(value):
            try:
                v = int(float(value))
            except Exception:
                v = int(morph_kernel_var.get())
            morph_kernel_label.configure(text=str(v))
            self.manual_settings["morph_kernel"] = v
            self.settings["morphology_kernel_size"] = v
        morph_kernel_slider.configure(command=update_morph_kernel_label)
        
        # 저장/취소 버튼
        button_frame = ctk.CTkFrame(content)
        button_frame.pack(pady=20)
        
        def save_params():
            self.easy_params["minG"] = int(minG_var.get())
            self.easy_params["ratG"] = float(ratG_var.get())
            self.easy_params["ratGb"] = float(ratGb_var.get())
            self.easy_params["minR"] = int(minR_var.get())
            self.easy_params["ratR"] = float(ratR_var.get())
            self.easy_params["dark_ratio_mult"] = float(dark_mult_var.get())
            self.easy_params["white_ratio_mult"] = float(white_mult_var.get())
            try:
                mk = int(morph_kernel_var.get())
                self.manual_settings["morph_kernel"] = mk
                self.settings["morphology_kernel_size"] = mk
            except Exception:
                pass
            
            # 수동 조정 플래그 설정 (자동 파라미터 추정 비활성화)
            self._user_manually_adjusted_params = True
            
            # 레이블 업데이트
            if hasattr(self, 'easy_params_label'):
                self.easy_params_label.configure(
                    text=f"G>{self.easy_params['minG']}, G/R>{self.easy_params['ratG']:.2f}, G/B>{self.easy_params['ratGb']:.2f}"
                )
            
            param_window.destroy()
            self._safe_refocus()
            messagebox.showinfo("성공", "파라미터가 저장되었습니다.\n\n기본 분석 시 수동 설정값을 사용합니다.")
            self._safe_refocus()
        
        def close_window():
            param_window.destroy()
            self._safe_refocus()
        
        ctk.CTkButton(button_frame, text="저장", command=save_params).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="취소", command=close_window).pack(side="left", padx=5)
