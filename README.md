
# Pauli Propagation Surrogate

## Overview
Pauli Propagation Surrogate is a research/experimental codebase for fast and efficient quantum simulation and Pauli operator-based computation using GPU-accelerated Python libraries such as cuQuantum, CuPy, and PyTorch.

## Features
- Pauli operator simulation using cuQuantum
- GPU-accelerated vector/matrix operations (CuPy, PyTorch)
- Example notebooks for quantum machine learning, VQE, and more
- Jupyter Notebook tutorials and ready-to-run scripts

## Project Structure

- `src/easy_cuQU.py`: Main Python module for easy Pauli propagation
- `cuQuantum/cuQu_example.ipynb`: Main usage example notebook
- `cuQuantum/my_tutorial_v2.ipynb`: Step-by-step tutorial
- `cuQuantum/truncation.ipynb`: Truncation strategies and memory optimization
- `cuQuantum/PYTHON_API_GUIDE.md`: cuQuantum Python API usage guide
- `Gradient_example/VQE_cuQu.ipynb`: VQE and gradient-based quantum algorithms
- `src/information.ipynb`: In-depth Pauli propagation guide
- `requirements.txt`: Full list of required Python packages
- `ENV_SETUP.sh`: Quick environment setup script (conda + pip)
- `LICENSE`: MIT License

## Installation
1. Prepare a conda environment and install all dependencies:
	```bash
	bash ENV_SETUP.sh
	```
	Or manual setup:
	```bash
	conda create -n cuQu python=3.11
	conda activate cuQu
	pip install -r requirements.txt
	```

2. Activate the environment:
	```bash
	conda activate cuQu
	```

## Usage
- Main script: `src/easy_cuQU.py`
- Example notebooks: `cuQuantum/cuQu_example.ipynb`, `cuQuantum/my_tutorial_v2.ipynb`
- Run example:
	```bash
	python src/easy_cuQU.py
	```

## Dependencies
Key packages:
- cuquantum-python-cu12
- cupy-cuda12x
- numpy, pandas, matplotlib, seaborn
- scipy, scikit-learn
- torch, torchvision, torchaudio (CUDA 12)
- jupyter, jupyterlab, ipykernel, ipywidgets
- tqdm, h5py, plotly, networkx, pillow
- pennylane, cudaq

See `requirements.txt` for the full list.

## License & Contact
- License: MIT License (see LICENSE file)
- Contact: kimhw7537@gmail.com

## Notes
- This code is intended for research and experimental use. For commercial use, please contact the author in advance.
