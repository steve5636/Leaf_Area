#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Leaf Area Analyzer - Main Entry Point
GrabCut 기반 전경(잎) 분리 + 형태학 분석
"""

import sys
import cv2
import numpy as np

from leaf_analyzer import AdvancedLeafAnalyzer

# CTK 가용성 확인을 위한 import
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False


def main():
    """메인 함수"""
    print("=" * 60)
    print("Advanced Leaf Area Analyzer v2.0")
    print("GrabCut 기반 전경(잎) 분리 + 형태학 분석")
    print("=" * 60)
    print()
    
    # 시스템 상태 확인
    print("시스템 상태 확인:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    
    # 기능 모듈 상태 확인
    features_status = []
    
    if CTK_AVAILABLE:
        features_status.append("✓ 현대적 UI (CustomTkinter)")
    else:
        features_status.append("△ 기본 UI (tkinter)")
    
    features_status.append("✓ GrabCut 세그멘테이션")
    
    for status in features_status:
        print(f"  {status}")
    
    print()
    print("주요 기능:")
    print("  • 다중 시드 기반 Color Picker")
    print("  • GrabCut 기반 세그멘테이션")
    print("  • 실시간 미리보기")
    print("  • 개별 객체 형태학적 분석 (OBB, 면적, 둘레 등)")
    print("  • 향상된 CSV/JSON 데이터 내보내기")
    print()
    
    print("애플리케이션을 시작합니다...")
    print("=" * 60)
    
    try:
        app = AdvancedLeafAnalyzer()
        app.run()
    except Exception as e:
        print(f"애플리케이션 시작 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
