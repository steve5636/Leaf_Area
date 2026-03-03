# Leaf Area Analyzer

식물 top-view 이미지에서 잎/스케일 면적을 빠르게 분석하는 GUI입니다.

중요: 이 저장소에는 `sam3/` 소스가 이미 포함되어 있습니다.  
`sam3`를 별도로 clone할 필요가 없습니다.

## 0) 저장소 클론
```bash
git clone https://github.com/steve5636/Leaf_Area.git Leaf_Area_Analyzer
cd Leaf_Area_Analyzer
```

## 1) Python 환경 생성 (3.12.x)
```bash
conda create -n leaf_area python=3.12 -y
conda activate leaf_area
```

## 2) 의존성 설치

### A) CPU 또는 macOS MPS 사용자
```bash
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

### B) NVIDIA CUDA GPU 사용자
1. 먼저 SAM3 공식 레포에서 본인 환경에 맞는 PyTorch CUDA 설치 명령을 확인하세요.  
   (이 링크는 `sam3` 별도 clone용이 아니라, CUDA용 PyTorch 설치 참고용입니다.)
   - https://github.com/facebookresearch/sam3
2. 해당 명령으로 `torch`, `torchvision` CUDA 빌드를 먼저 설치하세요.
3. 그 다음 이 레포 의존성을 설치하세요.

```bash
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

참고: 앱의 SAM3 디바이스 우선순위는 `CUDA -> MPS -> CPU`입니다.

## 3) SAM3 가중치 접근 설정
SAM3 모델 가중치는 Hugging Face에서 처음 실행 시 자동 다운로드됩니다.

```bash
hf auth login
```

가중치 저장소: https://huggingface.co/facebook/sam3

## 4) 실행
```bash
python main.py
```

## 5) 동작 확인 (선택)
```bash
python -c "import torch; print('cuda=', torch.cuda.is_available()); print('mps=', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

## 6) 리사이즈 추론
UI의 `추론 리사이즈 배율`을 사용하면 입력 이미지를 축소한 뒤 추론합니다.  
오버레이/내보내기 결과도 해당 배율 기준으로 반영됩니다.

## 7) 추가 문서
macOS MPS 관련 상세 안내: `docs/sam3_mps_setup.md`
