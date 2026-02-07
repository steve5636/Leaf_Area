#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SAM3_DIR="${ROOT_DIR}/sam3"
ENV_NAME="sam3_mps"

if ! command -v mamba >/dev/null 2>&1; then
  echo "mamba가 없습니다. mamba 설치 후 다시 실행해주세요."
  exit 1
fi

if [ ! -d "${SAM3_DIR}" ]; then
  echo "sam3 레포 클론..."
  git clone https://github.com/facebookresearch/sam3.git "${SAM3_DIR}"
fi

echo "PR #173 브랜치 체크아웃..."
cd "${SAM3_DIR}"
git fetch origin pull/173/head:device-agnostic
git checkout device-agnostic

echo "가상환경 확인..."
ENV_EXISTS="$(mamba env list --json | python - <<'PY'
import json, sys
data = json.load(sys.stdin)
paths = data.get("envs", [])
print("yes" if any(p.endswith("/sam3_mps") for p in paths) else "no")
PY
)"

if [ "${ENV_EXISTS}" = "no" ]; then
  echo "가상환경 생성: ${ENV_NAME}"
  mamba create -n "${ENV_NAME}" python=3.12 -y
fi

echo "sam3 editable 설치 및 의존성 설치..."
mamba run -n "${ENV_NAME}" python -m pip install -e .
mamba run -n "${ENV_NAME}" python -m pip install einops pycocotools

echo "완료. 아래 명령으로 실행하세요:"
echo "mamba activate ${ENV_NAME}"
echo "export PYTORCH_ENABLE_MPS_FALLBACK=1"
