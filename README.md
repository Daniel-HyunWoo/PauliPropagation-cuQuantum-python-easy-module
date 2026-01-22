
# Pauli Propagation Surrogate

## 프로젝트 개요
Pauli Propagation Surrogate는 cuQuantum, CuPy, PyTorch 등 GPU 가속 Python 라이브러리를 활용하여 양자 시뮬레이션 및 파울리 연산자 기반 계산을 빠르고 효율적으로 수행할 수 있도록 지원하는 연구/실험용 코드입니다.

## 주요 기능
- cuQuantum 기반 파울리 연산자 시뮬레이션
- GPU 가속 벡터/행렬 연산 (CuPy, PyTorch)
- 양자 머신러닝 및 VQE 등 예제 제공
- Jupyter Notebook 예제 및 튜토리얼 포함

## 설치 방법
1. Conda 환경 준비 및 패키지 설치
	```bash
	bash ENV_SETUP.sh
	```
	또는 수동 설치:
	```bash
	conda create -n cuQu python=3.11
	conda activate cuQu
	pip install -r requirements.txt
	```

2. 환경 활성화
	```bash
	conda activate cuQu
	```

## 사용법
- 주요 스크립트: `src/easy_cuQU.py`
- 예제 노트북: `cuQuantum/cuQu_example.ipynb`, `cuQuantum/my_tutorial_v2.ipynb`
- 실행 예시:
	```bash
	python src/easy_cuQU.py
	```

## 의존성
- cuquantum-python-cu12
- cupy-cuda12x
- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn
- torch, torchvision, torchaudio (CUDA 12)
- jupyter, jupyterlab, ipykernel, ipywidgets
- tqdm, h5py, plotly, networkx, pillow
- pennylane, cudaq

자세한 패키지 목록은 `requirements.txt` 참고

## 참고 및 문의
- 문의: [kimhw7537@gmail.com]
- 라이선스: MIT License (LICENSE 파일 참고)

## 기타
- 본 코드는 연구 및 실험 목적이며, 상업적 사용 전 별도 문의 바랍니다.
