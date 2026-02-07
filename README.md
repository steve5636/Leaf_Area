# Leaf Area Analyzer

잎 면적 분석 및 SAM3 기반 혼합 분석 GUI.

## 1) 설치 (conda)
```bash
conda create -n leaf_area python=3.12 -y
conda activate leaf_area
pip install -r requirements.txt
```

## 2) 실행
```bash
python main.py
```

## 3) SAM3 사용 (선택)
이 레포에는 `sam3/` 소스가 포함되어 있습니다.  
SAM3 모델 가중치는 Hugging Face 저장소에서 내려받습니다(접근 권한 필요).  
가중치 저장소: https://huggingface.co/facebook/sam3

### 설치 (공통)
```bash
pip install -e ./sam3
hf auth login
```

### macOS MPS (권장)
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

설치 가이드는 `docs/sam3_mps_setup.md` 참고.

## 4) 리사이즈 추론
UI의 `추론 리사이즈 배율`에 숫자를 입력하면,  
입력값 기준으로 이미지를 축소해 추론하고, 오버레이/내보내기도 그 해상도로 적용됩니다.

---
문의/개선 요청은 Issues로 남겨주세요.
