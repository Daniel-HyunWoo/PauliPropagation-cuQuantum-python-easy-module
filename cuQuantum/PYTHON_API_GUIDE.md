# cuPauliProp Python Bindings 사용 가이드

## 설치

최신 cuQuantum Python (24.08+)이 필요합니다:

## 주요 함수 설명

### 1. Library Management

```python
# Handle 생성 - 모든 함수 호출에 필요
handle = cupp.create()

# Stream 설정 (선택사항)
cupp.set_stream(handle, cuda_stream)

# Handle 삭제
cupp.destroy(handle)
```

### 2. Pauli Expansion 관리

```python
# Packed integer 개수 계산
# num_qubits=4 → 1개, num_qubits=128 → 2개
num_packed = cupp.get_num_packed_integers(num_qubits)

# Pauli expansion 생성
expansion = cupp.create_pauli_expansion(
    handle,
    num_qubits,           # 큐비트 개수
    pauli_buffer_ptr,     # GPU 메모리 포인터 (Pauli strings)
    pauli_buffer_size,    # Buffer 크기 (bytes)
    coef_buffer_ptr,      # GPU 메모리 포인터 (계수들)
    coef_buffer_size,     # Buffer 크기 (bytes)
    data_type,            # 1=CUDA_R_64F (float64)
    num_terms,            # 현재 term 개수
    is_sorted,            # 1=정렬됨, 0=아님
    is_unique             # 1=중복없음, 0=중복있음
)

# Term 개수 확인
n = cupp.pauli_expansion_get_num_terms(handle, expansion)

# View 생성 (range 선택)
view = cupp.pauli_expansion_get_contiguous_range(
    handle, expansion, start_idx, num_terms
)

# Expansion 삭제
cupp.destroy_pauli_expansion(expansion)
cupp.destroy_pauli_expansion_view(view)
```

### 3. Quantum Operators

```python
# Pauli Rotation Gate: exp(-i * angle/2 * P)
# P = X, Y, Z 또는 tensor product
qubits = np.array([0, 1], dtype=np.int32)
paulis = np.array([1, 2], dtype=np.int32)  # X⊗Y
operator = cupp.create_pauli_rotation_gate_operator(
    handle,
    angle,                  # 회전 각도 (라디안)
    num_qubits,            # gate가 작용하는 큐비트 수
    qubits.ctypes.data,    # 큐비트 인덱스
    paulis.ctypes.data     # Pauli 종류 (0=I, 1=X, 2=Y, 3=Z)
)

# Clifford Gate (CNOT, CZ, S, H 등)
qubits = np.array([control, target], dtype=np.int32)
operator = cupp.create_clifford_gate_operator(
    handle,
    gate_kind,             # 0=CX, 1=CY, 2=CZ, 3=S, 4=Sdg, ...
    qubits.ctypes.data     # 큐비트 인덱스
)

# Operator 삭제
cupp.destroy_operator(operator)
```

**Clifford Gate 종류:**
- `0`: CX (CNOT)
- `1`: CY
- `2`: CZ
- `3`: S
- `4`: S†
- `5`: H (Hadamard)
- `6`: X
- `7`: Y
- `8`: Z

### 4. Gate 적용 (Pauli Propagation 핵심!)

```python
# Operator를 Pauli expansion에 적용
cupp.pauli_expansion_view_compute_operator_application(
    handle,
    input_view,            # 입력 expansion view
    output_expansion,      # 출력 expansion (결과 저장)
    operator,              # 적용할 gate
    adjoint,               # 1=adjoint, 0=normal
    make_sorted,           # 1=출력 정렬, 0=안함
    keep_duplicates,       # 1=중복허용, 0=중복제거
    num_truncations,       # Truncation 전략 개수
    truncation_strategies, # Truncation 배열
    workspace             # Workspace descriptor
)
```

### 5. Truncation (메모리 절약)

```python
# 계수 기반 truncation
coef_trunc = cupp.CoefficientTruncationParams()
coef_trunc.cutoff = 1e-4  # |coefficient| < 1e-4 제거

# Pauli weight 기반 truncation  
weight_trunc = cupp.PauliWeightTruncationParams()
weight_trunc.cutoff = 8   # weight > 8 제거

# Truncation strategy 배열 생성
truncations = [
    cupp.TruncationStrategy(
        strategy_kind=0,  # COEFFICIENT_BASED
        params=coef_trunc
    ),
    cupp.TruncationStrategy(
        strategy_kind=1,  # PAULI_WEIGHT_BASED
        params=weight_trunc
    )
]
```

### 6. 기댓값 계산

```python
# Tr(view * |0⟩⟨0|) 계산
result = np.array([0.0], dtype=np.float64)
cupp.pauli_expansion_view_compute_trace_with_zero_state(
    handle,
    view,                  # Pauli expansion view
    result.ctypes.data,    # 결과 포인터
    workspace             # Workspace descriptor
)
expectation = result[0]

# 두 expansion의 trace
result = np.array([0.0], dtype=np.float64)
cupp.pauli_expansion_view_compute_trace_with_expansion_view(
    handle,
    view1,                 # 첫 번째 view
    view2,                 # 두 번째 view
    result.ctypes.data,    # 결과 포인터
    workspace             # Workspace descriptor
)
```

### 7. Workspace 관리

```python
# Workspace descriptor 생성
workspace = cupp.create_workspace_descriptor(handle)

# 메모리 할당 및 설정
size = 10 * 1024 * 1024  # 10 MB
d_buffer = cupy.cuda.alloc(size)
cupp.workspace_set_memory(
    handle,
    workspace,
    0,                     # MEMSPACE_DEVICE
    0,                     # WORKSPACE_SCRATCH
    d_buffer.ptr,          # 포인터
    size                   # 크기 (bytes)
)

# 필요한 크기 확인
required_size = cupp.workspace_get_memory_size(
    handle, workspace, 0, 0
)

# 삭제
cupp.destroy_workspace_descriptor(workspace)
```

## Pauli String 인코딩 방식

Pauli string은 두 개의 bit mask로 표현:
- **X mask**: X 또는 Y가 있는 위치에 1
- **Z mask**: Z 또는 Y가 있는 위치에 1

```python
# 예시: XYZI (4 qubits)
# Pauli:  X(0)  Y(1)  Z(2)  I(3)
# 위치:   bit0  bit1  bit2  bit3

X_mask = 0b0011  # X at 0, Y at 1
Z_mask = 0b0110  # Y at 1, Z at 2

# Packed array 형식: [X_mask, Z_mask]
pauli_packed = np.array([0b0011, 0b0110], dtype=np.uint64)
```

## 메모리 관리 팁

1. **Buffer 크기**: Term이 증가하므로 충분히 크게
2. **CuPy 사용**: GPU 메모리 관리에 편리
3. **Pointer 전달**: `.data.ptr` (CuPy) 또는 `.ctypes.data` (NumPy)

```python
# CuPy (GPU)
gpu_array = cupy.zeros(100, dtype=np.float64)
ptr = gpu_array.data.ptr

# NumPy (CPU)
cpu_array = np.zeros(100, dtype=np.int32)
ptr = cpu_array.ctypes.data
```

## 에러 처리

```python
try:
    handle = cupp.create()
    # ... operations ...
except Exception as e:
    print(f"Error: {e}")
    # cuPauliProp error code를 확인
    error_msg = cupp.get_error_string(error_code)
    print(f"cuPauliProp: {error_msg}")
finally:
    if handle:
        cupp.destroy(handle)
```

## 성능 최적화

1. **Truncation 사용**: 메모리와 속도 개선
2. **Stream 사용**: 비동기 실행
3. **Memory pool**: 반복 할당 피하기
4. **Sorted/Unique**: 가능하면 유지

## 참고 자료

- [cuPauliProp C API](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/index.html)
- [Python Bindings](https://docs.nvidia.com/cuda/cuquantum/latest/python/bindings/cupauliprop.html)
- [Kicked Ising Example](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/examples.html)
