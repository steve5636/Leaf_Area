# SAM3 MPS 설치 가이드 (macOS 전용)

이 문서는 **macOS + Apple Silicon(M1~M4)** 환경에서 SAM3를 **MPS fallback**으로 사용하기 위한 설치 가이드입니다.  
Windows CPU 사용자는 **지원 대상에서 제외**합니다.

## 대상
- macOS 13 이상
- Apple Silicon (M1/M2/M3/M4)
- Python 3.12
- `mamba` 사용 권장

## 설치 개요
1. `sam3` 레포를 PR #173 브랜치로 체크아웃
2. 가상환경 생성 및 패키지 설치
3. Hugging Face 로그인(체크포인트 다운로드)
4. MPS fallback 활성화 후 이미지 inference 테스트

## 설치 스크립트
아래 스크립트를 실행하면 **PR #173 적용 + 패키지 설치 + 의존성 설치**까지 자동으로 진행됩니다.

```
./scripts/setup_sam3_mps.sh
```

## Hugging Face 로그인 (체크포인트 다운로드)
```
hf auth login
```

## MPS fallback 활성화
```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 이미지 inference 테스트
아래 코드는 체크포인트를 자동으로 다운로드하고, 간단한 텍스트 프롬프트로 마스크를 생성합니다.

```python
import os
from PIL import Image
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = build_sam3_image_model(device=device, eval_mode=True, compile=False)
processor = Sam3Processor(model, device=device, confidence_threshold=0.4)

image = Image.open("IMG_0708.jpg").convert("RGB")
state = processor.set_image(image)
outputs = processor.set_text_prompt("leaf", state)
print(outputs.keys())
```

## 주의사항
- PR #173은 **아직 main에 머지되지 않았습니다**. 반드시 PR 브랜치를 사용하세요.
- MPS에서 일부 연산은 CPU fallback 경고가 발생할 수 있습니다(성능 영향).
- `sam3` 패키지 설치 중 `numpy` 버전이 내려가며 OpenCV 등과 충돌 경고가 발생할 수 있습니다.  
  충돌이 실제 문제를 일으키는 경우 별도 환경 분리를 권장합니다.
