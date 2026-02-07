# Leaf Area Analyzer

잎 면적 분석 및 SAM3 기반 혼합 분석 GUI.

## 1) 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 실행
```bash
python main.py
```

## 3) SAM3 사용 (선택)
SAM3 모델 가중치는 Hugging Face 저장소에서 내려받습니다. 접근 권한이 필요합니다.  
가중치 저장소: https://huggingface.co/facebook/sam3

### 설치 가이드 (macOS MPS)
`docs/sam3_mps_setup.md` 참고.

### 요약
```bash
# SAM3 레포 (PR #173 브랜치) 설치
./scripts/setup_sam3_mps.sh

# Hugging Face 로그인
hf auth login
```

## 4) 리사이즈 추론
UI의 `추론 리사이즈 배율`에 숫자를 입력하면,  
입력값 기준으로 이미지를 축소해 추론하고, 오버레이/내보내기도 그 해상도로 적용됩니다.

---
문의/개선 요청은 Issues로 남겨주세요.
