#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed Manager for Advanced Leaf Analyzer
시드 관리 클래스
"""

import numpy as np


class SeedManager:
    """시드 관리 클래스"""
    
    def __init__(self):
        self.seeds = {"leaf": [], "scale": []}  # background 제거
        self.current_class = "leaf"
        
    def add_seed(self, x: int, y: int, seed_class: str = None):
        """시드 추가"""
        if seed_class is None:
            seed_class = self.current_class
        self.seeds[seed_class].append((x, y))
    
    def remove_last_seed(self, seed_class: str = None):
        """마지막 시드 제거 (Undo)"""
        if seed_class is None:
            seed_class = self.current_class
        if self.seeds[seed_class]:
            self.seeds[seed_class].pop()
    
    def clear_seeds(self, seed_class: str = None):
        """시드 초기화"""
        if seed_class is None:
            self.seeds[self.current_class] = []
        else:
            self.seeds[seed_class] = []
    
    def remove_seed_at_position(self, x: int, y: int, seed_class: str = None, threshold: int = 15):
        """특정 위치 근처의 시드 제거 (우클릭 삭제용)"""
        if seed_class is None:
            seed_class = self.current_class
            
        if seed_class not in self.seeds:
            return False
            
        seeds = self.seeds[seed_class]
        if not seeds:
            return False
            
        # 클릭 위치에서 가장 가까운 시드 찾기
        min_distance = float('inf')
        closest_index = -1
        
        for i, (seed_x, seed_y) in enumerate(seeds):
            distance = ((x - seed_x) ** 2 + (y - seed_y) ** 2) ** 0.5
            if distance < min_distance and distance <= threshold:
                min_distance = distance
                closest_index = i
        
        # 임계값 이내의 시드가 있으면 제거
        if closest_index != -1:
            removed_seed = seeds.pop(closest_index)
            print(f"시드 제거: {removed_seed} (거리: {min_distance:.1f}px)")
            return True
        else:
            print(f"제거할 시드가 없습니다. (임계값: {threshold}px)")
            return False
    
    def get_patch_samples(self, image: np.ndarray, seed_class: str, patch_size: int = 11):
        """시드 주변 패치 샘플 추출"""
        if seed_class not in self.seeds or not self.seeds[seed_class]:
            return np.array([]).reshape(0, 3)
            
        patches = []
        half_size = patch_size // 2
        h, w = image.shape[:2]
        
        for x, y in self.seeds[seed_class]:
            # 경계 처리
            y1, y2 = max(0, y-half_size), min(h, y+half_size+1)
            x1, x2 = max(0, x-half_size), min(w, x+half_size+1)
            
            patch = image[y1:y2, x1:x2]
            patches.extend(patch.reshape(-1, 3))
        
        return np.array(patches)
