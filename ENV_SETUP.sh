#!/bin/bash

# 빠른 cuQuantum Python 환경 설치 (pip 기반)
# 환경 이름: cuQu

echo "========================================"
echo "Fast cuQuantum Setup for cuQu"
echo "========================================"
echo ""

# 1. 환경이 이미 있으면 제거
if conda env list | grep -q "^cuQu "; then
    echo "Removing existing cuQu environment..."
    conda env remove -n cuQu -y
fi

# 2. Python만으로 환경 생성 (빠름)
echo "Step 1: Creating minimal conda environment..."
conda create -n cuQu python=3.11 -y --quiet

if [ $? -ne 0 ]; then
    echo "✗ Failed"
    exit 1
fi
echo "✓ Base environment created"

# 3. 환경 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cuQu

# 4. pip로 핵심 패키지만 빠르게 설치
echo ""
echo "Step 2: Installing packages via pip (faster)..."
echo ""

echo "Installing cuQuantum Python..."
pip install --quiet cuquantum-python-cu12
echo "✓ cuQuantum Python"

echo "Installing CuPy..."
pip install --quiet cupy-cuda12x
echo "✓ CuPy"

echo "Installing NumPy, Pandas, Matplotlib..."
pip install --quiet numpy pandas matplotlib seaborn
echo "✓ NumPy, Pandas, Matplotlib, Seaborn"

echo "Installing SciPy, scikit-learn..."
pip install --quiet scipy scikit-learn
echo "✓ SciPy, scikit-learn"

echo "Installing PyTorch (CUDA 12)..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch"

echo "Installing Jupyter..."
pip install --quiet jupyter jupyterlab ipykernel ipywidgets
echo "✓ Jupyter"

echo "Installing additional tools..."
pip install --quiet tqdm h5py plotly networkx pillow
echo "✓ Additional packages"

echo "Installing PennyLane..."
pip install --quiet pennylane
echo "✓ PennyLane"

echo "Installing CUDA Quantum..."
pip install --quiet cudaq
echo "✓ CUDA Quantum"

# 5. 환경 정보 출력
echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""

python --version

echo ""
echo "Key packages:"
python -c "
import sys
try:
    import cuquantum
    print(f'✓ cuQuantum: {cuquantum.__version__}')
except: print('✗ cuQuantum')

try:
    import cupy
    print(f'✓ CuPy: {cupy.__version__}')
except: print('✗ CuPy')

try:
    import numpy
    print(f'✓ NumPy: {numpy.__version__}')
except: print('✗ NumPy')

try:
    import pandas
    print(f'✓ Pandas: {pandas.__version__}')
except: print('✗ Pandas')

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
except: print('✗ PyTorch')

try:
    import matplotlib
    print(f'✓ Matplotlib: {matplotlib.__version__}')
except: print('✗ Matplotlib')

try:
    import scipy
    print(f'✓ SciPy: {scipy.__version__}')
except: print('✗ SciPy')

try:
    import sklearn
    print(f'✓ scikit-learn: {sklearn.__version__}')
except: print('✗ scikit-learn')

try:
    import pennylane
    print(f'✓ PennyLane: {pennylane.__version__}')
except: print('✗ PennyLane')

try:
    import cudaq
    print(f'✓ CUDA Quantum: {cudaq.__version__}')
except: print('✗ CUDA Quantum')

try:
    import jupyter
    print(f'✓ Jupyter: installed')
except: print('✗ Jupyter')
" 2>/dev/null

echo ""
echo "========================================"
echo "✓ Setup completed!"
echo "========================================"
echo ""
echo "To use:"
echo "  conda activate cuQu"
echo "  cd /home/ubuntu/latent/CUDA-Q"
echo "  python simple_pauli_prop.py"
echo ""
