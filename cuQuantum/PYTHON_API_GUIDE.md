# cuPauliProp Python Bindings Usage Guide

## Installation

Requires cuQuantum Python (24.08+) or later:

## Main Function Descriptions

### 1. Library Management

```python
# Create a handle - required for all function calls
handle = cupp.create()

# Set stream (optional)
cupp.set_stream(handle, cuda_stream)

# Destroy handle
cupp.destroy(handle)
```

### 2. Pauli Expansion Management

```python
# Calculate number of packed integers
# num_qubits=4 → 1, num_qubits=128 → 2
num_packed = cupp.get_num_packed_integers(num_qubits)

# Create Pauli expansion
expansion = cupp.create_pauli_expansion(
    handle,
    num_qubits,           # Number of qubits
    pauli_buffer_ptr,     # GPU memory pointer (Pauli strings)
    pauli_buffer_size,    # Buffer size (bytes)
    coef_buffer_ptr,      # GPU memory pointer (coefficients)
    coef_buffer_size,     # Buffer size (bytes)
    data_type,            # 1=CUDA_R_64F (float64)
    num_terms,            # Current number of terms
    is_sorted,            # 1=sorted, 0=not sorted
    is_unique             # 1=unique, 0=not unique
)

# Check number of terms
n = cupp.pauli_expansion_get_num_terms(handle, expansion)

# Create view (select range)
view = cupp.pauli_expansion_get_contiguous_range(
    handle, expansion, start_idx, num_terms
)

# Destroy expansion
cupp.destroy_pauli_expansion(expansion)
cupp.destroy_pauli_expansion_view(view)
```

### 3. Quantum Operators

#### Gate Types

##### 1. Pauli Rotation Gate
- Form: $e^{-i\theta P/2}$, P is a Pauli operator
- API: `create_pauli_rotation_gate_operator`
- Pauli types: 0=I, 1=X, 2=Z, 3=Y

##### 2. Clifford Gate
- CNOT, CZ, H, S, etc.
- API: `create_clifford_gate_operator`
- Gate types:
    - 0: I (CNOT)
    - 1: X
    - 2: Z
    - 3: Y
    - 4: H
    - 5: S
    - 7: CX [target, control] order
    - 8: CZ
    - 9: CY
    - 10: SWAP
    - 11: ISWAP
    - 12: SQRTX
    - 13: SQRTZ
    - 14: SQRTY

```python
# Pauli Rotation Gate: exp(-i * angle/2 * P)
# P = X, Y, Z or tensor product
qubits = np.array([0, 1], dtype=np.int32)
paulis = np.array([1, 2], dtype=np.int32)  # X⊗Y
operator = cupp.create_pauli_rotation_gate_operator(
    handle,
    angle,                  # 회전 각도 (라디안)
    num_qubits,            # gate가 작용하는 큐비트 수
    qubits.ctypes.data,    # 큐비트 인덱스
    paulis.ctypes.data     

# Clifford Gate (CNOT, CZ, S, H, etc.)
qubits = np.array([control, target], dtype=np.int32)
operator = cupp.create_clifford_gate_operator(
    handle,
    gate_kind,             # 0=CX, 1=CY, 2=CZ, 3=S, 4=Sdg, ...
    qubits.ctypes.data     # 큐비트 인덱스
)

# Destroy operator
cupp.destroy_operator(operator)
```
---

#### Hadamard Decomposition
H = RZ(π/2) RY(π/2) RZ(π/2)

### 4. Gate Application (Core of Pauli Propagation!)

```python
# Apply operator to Pauli expansion
cupp.pauli_expansion_view_compute_operator_application(
    handle,
    input_view,            # 입력 expansion view
    output_expansion,      # 출력 expansion (결과 저장)
    operator,              # 적용할 gate
    adjoint,               # 1=adjoint, 0=normal
    make_sorted,           # 1=output sorted, 0=not sorted
    keep_duplicates,       # 1=allow duplicates, 0=remove duplicates
    num_truncations,       # Number of truncation strategies
    truncation_strategies, # Truncation array
    workspace             # Workspace descriptor
)
```

### 5. Truncation (Memory Saving)

```python
# Coefficient-based truncation
coef_trunc = cupp.CoefficientTruncationParams()
coef_trunc.cutoff = 1e-4  # |coefficient| < 1e-4 제거

# Pauli weight-based truncation  
weight_trunc = cupp.PauliWeightTruncationParams()
weight_trunc.cutoff = 8   # weight > 8 제거

# Create truncation strategy array
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

### 6. Expectation Value Calculation

```python
# Compute Tr(view * |0⟩⟨0|)
result = np.array([0.0], dtype=np.float64)
cupp.pauli_expansion_view_compute_trace_with_zero_state(
    handle,
    view,                  # Pauli expansion view
    result.ctypes.data,    # 결과 포인터
    workspace             # Workspace descriptor
)
expectation = result[0]

# Trace of two expansions
result = np.array([0.0], dtype=np.float64)
cupp.pauli_expansion_view_compute_trace_with_expansion_view(
    handle,
    view1,                 # 첫 번째 view
    view2,                 # 두 번째 view
    result.ctypes.data,    # 결과 포인터
    workspace             # Workspace descriptor
)
```

### 7. Workspace Management

```python
# Create workspace descriptor
workspace = cupp.create_workspace_descriptor(handle)

# Allocate and set memory
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

# Check required size
required_size = cupp.workspace_get_memory_size(
    handle, workspace, 0, 0
)

# Delete
cupp.destroy_workspace_descriptor(workspace)
```

## Pauli String Encoding Method

Pauli strings are represented by two bit masks:
- **X mask**: 1 where X or Y is present
- **Z mask**: 1 where Z or Y is present

```python
# Example: XYZI (4 qubits)
# Pauli:  X(0)  Y(1)  Z(2)  I(3)
# Position: bit0  bit1  bit2  bit3

X_mask = 0b0011  # X at 0, Y at 1
Z_mask = 0b0110  # Y at 1, Z at 2

# Packed array format: [X_mask, Z_mask]
pauli_packed = np.array([0b0011, 0b0110], dtype=np.uint64)
```

## Memory Management Tips

1. **Buffer size**: Make it large enough as the number of terms increases
2. **Use CuPy**: Convenient for GPU memory management
3. **Pointer passing**: Use `.data.ptr` (CuPy) or `.ctypes.data` (NumPy)

```python
# CuPy (GPU)
gpu_array = cupy.zeros(100, dtype=np.float64)
ptr = gpu_array.data.ptr

# NumPy (CPU)
cpu_array = np.zeros(100, dtype=np.int32)
ptr = cpu_array.ctypes.data
```

## Error Handling

```python
try:
    handle = cupp.create()
    # ... operations ...
except Exception as e:
    print(f"Error: {e}")
    # Check cuPauliProp error code
    error_msg = cupp.get_error_string(error_code)
    print(f"cuPauliProp: {error_msg}")
finally:
    if handle:
        cupp.destroy(handle)
```

## Performance Optimization

1. **Use truncation**: Improve memory and speed
2. **Use stream**: Asynchronous execution
3. **Memory pool**: Avoid repeated allocation
4. **Sorted/Unique**: Keep if possible

## References

- [cuPauliProp C API](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/index.html)
- [Python Bindings](https://docs.nvidia.com/cuda/cuquantum/latest/python/bindings/cupauliprop.html)
- [Kicked Ising Example](https://docs.nvidia.com/cuda/cuquantum/latest/cupauliprop/examples.html)
