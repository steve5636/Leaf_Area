#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
색상 분석 유틸리티 모듈
HSV 원형 통계, CIELAB ΔE2000, K-means 최적화 등의 고급 색상 분석 기능
"""

import numpy as np
import cv2
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math


class CircularStatistics:
    """원형 통계 계산 클래스"""
    
    @staticmethod
    def circular_mean(angles_deg, weights=None):
        """원형 평균 계산 (도 단위)"""
        angles_rad = np.deg2rad(angles_deg)
        
        if weights is None:
            weights = np.ones(len(angles_deg))
        
        # 가중 평균 계산
        x = np.sum(weights * np.cos(angles_rad))
        y = np.sum(weights * np.sin(angles_rad))
        
        mean_rad = np.arctan2(y, x)
        return np.rad2deg(mean_rad) % 360
    
    @staticmethod
    def circular_std(angles_deg, weights=None):
        """원형 표준편차 계산 (도 단위)"""
        angles_rad = np.deg2rad(angles_deg)
        
        if weights is None:
            weights = np.ones(len(angles_deg))
        
        # R 계산 (평균 방향 벡터의 길이)
        x = np.sum(weights * np.cos(angles_rad)) / np.sum(weights)
        y = np.sum(weights * np.sin(angles_rad)) / np.sum(weights)
        R = np.sqrt(x**2 + y**2)
        
        # 원형 표준편차
        if R >= 1.0:
            return 0.0
        else:
            circular_var = 1 - R
            return np.rad2deg(np.sqrt(2 * circular_var))
    
    @staticmethod
    def circular_distance(angle1_deg, angle2_deg):
        """두 각도 간의 원형 거리 계산"""
        diff = abs(angle1_deg - angle2_deg)
        return min(diff, 360 - diff)


class RobustStatistics:
    """강건한 통계 계산 클래스"""
    
    @staticmethod
    def median_absolute_deviation(data, scale_factor=1.4826):
        """중앙절대편차 (MAD) 계산"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return mad * scale_factor  # 정규분포 가정 하에서 표준편차와 일치하도록 스케일링
    
    @staticmethod
    def robust_mean_std(data, outlier_threshold=2.0):
        """이상치를 제거한 강건한 평균과 표준편차"""
        median = np.median(data)
        mad = RobustStatistics.median_absolute_deviation(data)
        
        # 이상치 마스크
        outlier_mask = np.abs(data - median) <= outlier_threshold * mad
        
        if np.sum(outlier_mask) > 0:
            clean_data = data[outlier_mask]
            return np.mean(clean_data), np.std(clean_data)
        else:
            return median, mad


class DeltaE2000:
    """정확한 CIE ΔE2000 색차 계산 클래스"""
    
    @staticmethod
    def delta_e_2000(lab1, lab2):
        """
        CIE ΔE2000 색차 계산
        lab1, lab2: [L*, a*, b*] 형태의 Lab 색상값
        """
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2
        
        # 크로마 계산
        C1 = math.sqrt(a1**2 + b1**2)
        C2 = math.sqrt(a2**2 + b2**2)
        C_bar = (C1 + C2) / 2
        
        # G 계산
        G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25**7)))
        
        # a' 계산
        a1_prime = a1 * (1 + G)
        a2_prime = a2 * (1 + G)
        
        # C' 및 h' 계산
        C1_prime = math.sqrt(a1_prime**2 + b1**2)
        C2_prime = math.sqrt(a2_prime**2 + b2**2)
        
        h1_prime = math.atan2(b1, a1_prime) * 180 / math.pi
        if h1_prime < 0:
            h1_prime += 360
            
        h2_prime = math.atan2(b2, a2_prime) * 180 / math.pi
        if h2_prime < 0:
            h2_prime += 360
        
        # ΔL', ΔC', Δh' 계산
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        
        delta_h_prime = h2_prime - h1_prime
        if abs(delta_h_prime) > 180:
            if h2_prime > h1_prime:
                delta_h_prime -= 360
            else:
                delta_h_prime += 360
        
        delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
        
        # 평균값들
        L_bar_prime = (L1 + L2) / 2
        C_bar_prime = (C1_prime + C2_prime) / 2
        
        if abs(h1_prime - h2_prime) > 180:
            H_bar_prime = (h1_prime + h2_prime + 360) / 2
        else:
            H_bar_prime = (h1_prime + h2_prime) / 2
        
        # T 계산
        T = (1 - 0.17 * math.cos(math.radians(H_bar_prime - 30)) +
             0.24 * math.cos(math.radians(2 * H_bar_prime)) +
             0.32 * math.cos(math.radians(3 * H_bar_prime + 6)) -
             0.20 * math.cos(math.radians(4 * H_bar_prime - 63)))
        
        # 회전 항 (RT)
        delta_theta = 30 * math.exp(-(((H_bar_prime - 275) / 25) ** 2))
        RC = 2 * math.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
        RT = -math.sin(math.radians(2 * delta_theta)) * RC
        
        # 가중 함수들
        SL = 1 + (0.015 * (L_bar_prime - 50)**2) / math.sqrt(20 + (L_bar_prime - 50)**2)
        SC = 1 + 0.045 * C_bar_prime
        SH = 1 + 0.015 * C_bar_prime * T
        
        # 최종 ΔE2000 계산
        delta_E = math.sqrt(
            (delta_L_prime / SL)**2 +
            (delta_C_prime / SC)**2 +
            (delta_H_prime / SH)**2 +
            RT * (delta_C_prime / SC) * (delta_H_prime / SH)
        )
        
        return delta_E
    
    @staticmethod
    def delta_e_2000_vectorized(lab_center, lab_image):
        """벡터화된 ΔE2000 계산 (이미지 전체 대상)"""
        h, w, _ = lab_image.shape
        result = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                result[i, j] = DeltaE2000.delta_e_2000(lab_center, lab_image[i, j])
        
        return result


class OptimalClusters:
    """최적 군집 수 결정 클래스"""
    
    @staticmethod
    def elbow_method(data, max_k=10, random_state=42):
        """엘보우 메소드로 최적 K 결정"""
        if len(data) < 4:
            return 1
            
        inertias = []
        K_range = range(1, min(max_k + 1, len(data)))
        
        for k in K_range:
            if k == 1:
                inertias.append(np.var(data) * len(data))
            else:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
        
        # 엘보우 포인트 찾기
        if len(inertias) < 3:
            return 1
            
        # 2차 차분을 이용한 엘보우 포인트 탐지
        second_diff = np.diff(np.diff(inertias))
        if len(second_diff) > 0:
            elbow_idx = np.argmax(second_diff) + 2
            return K_range[elbow_idx] if elbow_idx < len(K_range) else K_range[-1]
        
        return 2
    
    @staticmethod
    def silhouette_method(data, max_k=10, random_state=42):
        """실루엣 스코어로 최적 K 결정"""
        if len(data) < 4:
            return 1
            
        best_score = -1
        best_k = 1
        K_range = range(2, min(max_k + 1, len(data)))
        
        for k in K_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = kmeans.fit_predict(data)
                score = silhouette_score(data, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        return best_k
    
    @staticmethod
    def combined_method(data, max_k=10, random_state=42):
        """엘보우와 실루엣 방법을 결합한 최적 K 결정"""
        elbow_k = OptimalClusters.elbow_method(data, max_k, random_state)
        silhouette_k = OptimalClusters.silhouette_method(data, max_k, random_state)
        
        # 두 방법의 결과가 비슷하면 그 값을 사용, 아니면 보수적으로 더 작은 값 선택
        if abs(elbow_k - silhouette_k) <= 1:
            return max(elbow_k, silhouette_k)
        else:
            return min(elbow_k, silhouette_k)


class AdaptiveTuning:
    """적응형 허용오차 튜닝 클래스"""
    
    def __init__(self, target_area_ratio=(0.003, 0.4), max_iterations=20):
        """
        target_area_ratio: 목표 면적 비율 범위 (최소, 최대)
        max_iterations: 최대 반복 횟수
        """
        self.target_min, self.target_max = target_area_ratio
        self.max_iterations = max_iterations
    
    def proportional_control(self, current_ratio, target_ratio, current_param, 
                           param_range, kp=0.5, max_step=0.1):
        """비례 제어 기반 파라미터 조정"""
        error = target_ratio - current_ratio
        
        # 정규화된 오차 (-1 ~ 1)
        if error > 0:  # 면적이 부족한 경우
            normalized_error = min(error / target_ratio, 1.0)
        else:  # 면적이 과다한 경우
            normalized_error = max(error / target_ratio, -1.0)
        
        # 조정량 계산
        param_min, param_max = param_range
        param_span = param_max - param_min
        
        adjustment = kp * normalized_error * param_span * max_step
        
        # 새로운 파라미터 값
        new_param = np.clip(current_param + adjustment, param_min, param_max)
        
        return new_param
    
    def tune_hsv_parameters(self, color_model, hsv_image, 
                          initial_params=None, verbose=False):
        """HSV 파라미터 자동 튜닝"""
        if initial_params is None:
            initial_params = color_model.hsv_params.copy()
        
        current_params = initial_params.copy()
        h, w = hsv_image.shape[:2]
        total_pixels = h * w
        
        for iteration in range(self.max_iterations):
            # 현재 파라미터로 마스크 생성
            color_model.hsv_params = current_params
            mask = color_model.predict_hsv(hsv_image)
            current_area = np.sum(mask)
            current_ratio = current_area / total_pixels
            
            if verbose:
                print(f"Iteration {iteration+1}: Area ratio = {current_ratio:.4f}")
            
            # 수렴 확인
            if self.target_min <= current_ratio <= self.target_max:
                if verbose:
                    print(f"Converged at iteration {iteration+1}")
                break
            
            # 목표 비율 (중간값)
            target_ratio = (self.target_min + self.target_max) / 2
            
            # 파라미터별 순차 조정
            if current_ratio < target_ratio:  # 면적 증가 필요
                # H 허용오차 증가
                current_params["h_tolerance"] = self.proportional_control(
                    current_ratio, target_ratio, current_params["h_tolerance"],
                    (5, 50), kp=0.3
                )
                
                # S 허용오차 증가  
                current_params["s_tolerance"] = self.proportional_control(
                    current_ratio, target_ratio, current_params["s_tolerance"],
                    (10, 100), kp=0.3
                )
                
                # V 허용오차 증가
                current_params["v_tolerance"] = self.proportional_control(
                    current_ratio, target_ratio, current_params["v_tolerance"],
                    (15, 120), kp=0.3
                )
                
            else:  # 면적 감소 필요
                # 반대 방향으로 조정
                current_params["v_tolerance"] = self.proportional_control(
                    current_ratio, target_ratio, current_params["v_tolerance"],
                    (15, 120), kp=0.3
                )
                
                current_params["s_tolerance"] = self.proportional_control(
                    current_ratio, target_ratio, current_params["s_tolerance"],
                    (10, 100), kp=0.3
                )
                
                current_params["h_tolerance"] = self.proportional_control(
                    current_ratio, target_ratio, current_params["h_tolerance"],
                    (5, 50), kp=0.3
                )
        
        return current_params


class BackgroundSuppression:
    """배경 억제 로직 클래스"""
    
    @staticmethod
    def absolute_background_mask(hsv_image, lab_image):
        """절대 기준 기반 배경 마스크"""
        h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
        
        # 저채도 + 극단 명도
        low_sat = s < 10
        extreme_bright = v > 245  # 거의 흰색
        extreme_dark = v < 10     # 거의 검은색
        
        background_mask = low_sat & (extreme_bright | extreme_dark)
        
        return background_mask
    
    @staticmethod
    def kmeans_background_detection(lab_image, leaf_seeds_lab, n_clusters=4):
        """K-means 기반 배경 영역 탐지"""
        h, w, c = lab_image.shape
        
        # 저해상도로 K-means 수행 (속도 최적화)
        scale_factor = 4
        small_lab = cv2.resize(lab_image, (w//scale_factor, h//scale_factor))
        lab_pixels = small_lab.reshape(-1, 3)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(lab_pixels)
        cluster_centers = kmeans.cluster_centers_
        
        # 잎 시드와 가장 거리가 먼 클러스터 찾기
        if leaf_seeds_lab is not None and len(leaf_seeds_lab) > 0:
            leaf_center = np.mean(leaf_seeds_lab, axis=0)
            
            distances = []
            cluster_sizes = []
            
            for i, center in enumerate(cluster_centers):
                # 잎 중심과의 거리
                dist = DeltaE2000.delta_e_2000(leaf_center, center)
                distances.append(dist)
                
                # 클러스터 크기 (픽셀 수)
                cluster_size = np.sum(cluster_labels == i)
                cluster_sizes.append(cluster_size)
            
            distances = np.array(distances)
            cluster_sizes = np.array(cluster_sizes)
            
            # 거리가 멀고 크기가 큰 클러스터를 배경으로 선택
            # 정규화된 점수 계산
            norm_dist = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
            norm_size = (cluster_sizes - cluster_sizes.min()) / (cluster_sizes.max() - cluster_sizes.min() + 1e-8)
            
            combined_score = 0.7 * norm_dist + 0.3 * norm_size
            background_cluster = np.argmax(combined_score)
            
            # 배경 마스크 생성
            background_mask_small = (cluster_labels == background_cluster).reshape(small_lab.shape[:2])
            background_mask = cv2.resize(
                background_mask_small.astype(np.uint8), (w, h), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            return background_mask
        
        return np.zeros((h, w), dtype=bool)


def main():
    """테스트 함수"""
    # 원형 통계 테스트
    angles = [350, 10, 20, 5, 355]
    print(f"원형 평균: {CircularStatistics.circular_mean(angles):.2f}도")
    print(f"원형 표준편차: {CircularStatistics.circular_std(angles):.2f}도")
    
    # ΔE2000 테스트
    lab1 = [50, 10, 20]
    lab2 = [55, 15, 25]
    delta_e = DeltaE2000.delta_e_2000(lab1, lab2)
    print(f"ΔE2000: {delta_e:.2f}")


if __name__ == "__main__":
    main()
