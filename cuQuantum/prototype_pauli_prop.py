"""
cuPauliProp Python Example - Pauli Propagation을 이용한 기댓값 계산

이 예제는 다음을 보여줍니다:
1. Observable을 Pauli expansion으로 초기화
2. Quantum circuit 구성 (Hadamard, CNOT, Rotation)
3. Heisenberg picture에서 Pauli propagation 수행
4. |0⟩ 상태에서 기댓값 계산

주의: cuQuantum Python 24.08+ 버전이 필요합니다
"""

import numpy as np
import cupy as cp
from cuquantum.bindings import cupauliprop as cupp

print("=" * 60)
print("cuPauliProp Python Example")
print("=" * 60)
print()

# ============================================================================
# 1. Library Setup & Handle 생성
# ============================================================================
print("1. Initializing cuPauliProp library...")

# Handle 생성 - 모든 cuPauliProp 함수 호출에 필요
# C의 cupaulipropCreate()에 해당
handle = cupp.create()
print("   ✓ Handle created")

# ============================================================================
# 2. Circuit 파라미터 설정
# ============================================================================
num_qubits = 4
print(f"\n2. Circuit parameters:")
print(f"   Number of qubits: {num_qubits}")

# Packed integer 개수 계산
# Pauli string (X, Z bits)을 저장하는데 필요한 64-bit integer 개수
# 예: 4 qubits → 1개, 128 qubits → 2개
num_packed_ints = cupp.get_num_packed_integers(num_qubits)
print(f"   Packed integers needed: {num_packed_ints}")

# ============================================================================
# 3. 초기 Observable 준비: Z_0
# ============================================================================
print(f"\n3. Preparing initial observable: Z_0")

# Pauli string을 packed integer로 인코딩
# 각 Pauli string은 X mask와 Z mask로 표현됨
# - X mask: X 또는 Y가 있는 위치에 1
# - Z mask: Z 또는 Y가 있는 위치에 1
# 예: Z on qubit 0 → X_mask=0b0000, Z_mask=0b0001

# Host에서 준비
initial_pauli_host = np.zeros(2 * num_packed_ints, dtype=np.uint64)
initial_pauli_host[num_packed_ints] = 0b0001  # Z mask: qubit 0에 Z
initial_coef_host = np.array([1.0], dtype=np.float64)

# Device로 복사 (cuPauliProp는 GPU에서 작동)
d_input_pauli = cp.asarray(initial_pauli_host)
d_input_coef = cp.asarray(initial_coef_host)

print(f"   Observable encoded as packed integers:")
print(f"   - X mask: {hex(initial_pauli_host[0])}")
print(f"   - Z mask: {hex(initial_pauli_host[num_packed_ints])}")
print(f"   - Coefficient: {initial_coef_host[0]}")

# ============================================================================
# 4. Pauli Expansion 생성
# ============================================================================
print(f"\n4. Creating Pauli expansions...")

# 입력용 expansion: 초기 observable 포함
# 충분한 메모리를 할당 (propagation 중 term이 증가하므로)
max_terms = 100
pauli_buffer_size = 2 * num_packed_ints * max_terms * 8  # bytes
coef_buffer_size = max_terms * 8  # bytes (float64)

# Buffer 확장 (더 많은 term을 위해)
d_input_pauli_buffer = cp.zeros(2 * num_packed_ints * max_terms, dtype=np.uint64)
d_input_pauli_buffer[:len(d_input_pauli)] = d_input_pauli
d_input_coef_buffer = cp.zeros(max_terms, dtype=np.float64)
d_input_coef_buffer[:len(d_input_coef)] = d_input_coef

# create_pauli_expansion() 파라미터:
# - handle: cuPauliProp handle
# - num_qubits: 큐비트 개수
# - pauli_buffer: Pauli string들을 저장할 GPU 메모리 (포인터)
# - pauli_buffer_size: Pauli buffer 크기 (bytes)
# - coef_buffer: 계수들을 저장할 GPU 메모리
# - coef_buffer_size: 계수 buffer 크기 (bytes)
# - data_type: 계수의 데이터 타입 (CUDA_R_64F = float64)
# - num_terms: 현재 포함된 term 개수
# - is_sorted: Pauli string이 정렬되어 있는지 (1=yes, 0=no)
# - is_unique: 중복 term이 없는지 (1=yes, 0=no)
input_expansion = cupp.create_pauli_expansion(
    handle,
    num_qubits,
    d_input_pauli_buffer.data.ptr,
    pauli_buffer_size,
    d_input_coef_buffer.data.ptr,
    coef_buffer_size,
    1,  # CUDA_R_64F (float64)
    1,  # 초기에 1개 term
    1,  # sorted
    1   # unique
)
print(f"   ✓ Input expansion created (capacity: {max_terms} terms)")

# 출력용 expansion: 처음엔 비어있음
d_output_pauli_buffer = cp.zeros(2 * num_packed_ints * max_terms, dtype=np.uint64)
d_output_coef_buffer = cp.zeros(max_terms, dtype=np.float64)

output_expansion = cupp.create_pauli_expansion(
    handle,
    num_qubits,
    d_output_pauli_buffer.data.ptr,
    pauli_buffer_size,
    d_output_coef_buffer.data.ptr,
    coef_buffer_size,
    1,  # CUDA_R_64F
    0,  # 초기에 0개 term
    0, 0
)
print(f"   ✓ Output expansion created")

# ============================================================================
# 5. Workspace 준비
# ============================================================================
print(f"\n5. Setting up workspace...")

# Workspace: cuPauliProp 연산에 필요한 임시 메모리
workspace_size = 10 * 1024 * 1024  # 10 MB
d_workspace = cp.cuda.alloc(workspace_size)

# workspace_descriptor 생성
workspace = cupp.create_workspace_descriptor(handle)

# workspace_set_memory() 파라미터:
# - handle: cuPauliProp handle
# - workspace: workspace descriptor
# - memspace: 메모리 위치 (0=device, 1=host)
# - kind: workspace 종류 (0=scratch temporary memory)
# - ptr: 메모리 포인터
# - size: 메모리 크기 (bytes)
cupp.workspace_set_memory(
    handle,
    workspace,
    0,  # CUPAULIPROP_MEMSPACE_DEVICE
    0,  # CUPAULIPROP_WORKSPACE_SCRATCH
    d_workspace.ptr,
    workspace_size
)
print(f"   ✓ Workspace allocated: {workspace_size / 1024 / 1024:.1f} MB")

# ============================================================================
# 6. Quantum Circuit 구성
# ============================================================================
print(f"\n6. Building quantum circuit...")

circuit = []
PI = np.pi

# (a) Hadamard on qubit 0
# H = RZ(π/2) RY(π/2) RZ(π/2) 로 분해
print("   Adding gates:")
print("   - H(0) = RZ(π/2) RY(π/2) RZ(π/2)")

qubit_0 = np.array([0], dtype=np.int32)
pauli_z = np.array([2], dtype=np.int32)  # CUPAULIPROP_PAULI_Z = 2
pauli_y = np.array([1], dtype=np.int32)  # CUPAULIPROP_PAULI_Y = 1

# create_pauli_rotation_gate_operator() 파라미터:
# - handle: cuPauliProp handle
# - angle: 회전 각도 (라디안)
# - num_qubits: 이 gate가 작용하는 큐비트 개수
# - qubits: 큐비트 인덱스 배열
# - paulis: Pauli 종류 배열 (0=I, 1=X, 2=Y, 3=Z)
# 반환: operator handle
rz1 = cupp.create_pauli_rotation_gate_operator(
    handle, PI/2, 1, qubit_0.ctypes.data, pauli_z.ctypes.data
)
ry = cupp.create_pauli_rotation_gate_operator(
    handle, PI/2, 1, qubit_0.ctypes.data, pauli_y.ctypes.data
)
rz2 = cupp.create_pauli_rotation_gate_operator(
    handle, PI/2, 1, qubit_0.ctypes.data, pauli_z.ctypes.data
)
circuit.extend([rz1, ry, rz2])

# (b) CNOT(0, 1)
print("   - CNOT(0, 1)")

cnot_qubits = np.array([0, 1], dtype=np.int32)

# create_clifford_gate_operator() 파라미터:
# - handle: cuPauliProp handle
# - gate_kind: Clifford gate 종류 (0=CX/CNOT, 1=CY, 2=CZ, 3=S, 등)
# - qubits: 큐비트 인덱스 배열 (2-qubit gate면 [control, target])
# 주의: 반환값(operator)을 받아서 circuit 리스트에 추가해야 함
cnot_op = cupp.create_clifford_gate_operator(
    handle,
    0,  # CUPAULIPROP_CLIFFORD_GATE_KIND_CX
    cnot_qubits.ctypes.data
)
circuit.append(cnot_op)

# (c) RY(π/4) on qubit 2
print("   - RY(π/4) on qubit 2")

qubit_2 = np.array([2], dtype=np.int32)
ry2 = cupp.create_pauli_rotation_gate_operator(
    handle, PI/4, 1, qubit_2.ctypes.data, pauli_y.ctypes.data
)
circuit.append(ry2)

print(f"\n   Total gates: {len(circuit)}")

# ============================================================================
# 7. Pauli Propagation (Heisenberg Picture)
# ============================================================================
print(f"\n7. Performing Pauli propagation...")
print("   (Propagating observable backwards through circuit)")

current_input = input_expansion
current_output = output_expansion

# Circuit을 역순으로 적용 (adjoint)
for i in range(len(circuit) - 1, -1, -1):
    gate = circuit[i]
    
    # 현재 expansion의 term 개수 확인
    num_terms = cupp.pauli_expansion_get_num_terms(handle, current_input)
    
    # View 생성: expansion의 일부 또는 전체를 가리키는 포인터
    # get_contiguous_range() 파라미터:
    # - handle: cuPauliProp handle
    # - expansion: Pauli expansion
    # - start_index: 시작 term 인덱스
    # - num_terms: 포함할 term 개수
    # 반환: view handle
    input_view = cupp.pauli_expansion_get_contiguous_range(
        handle, current_input, 0, num_terms
    )
    
    # pauli_expansion_view_compute_operator_application() 파라미터:
    # - handle: cuPauliProp handle
    # - input_view: 입력 Pauli expansion view
    # - output_expansion: 출력 Pauli expansion (여기에 결과 저장)
    # - operator: 적용할 quantum operator
    # - adjoint: adjoint 적용 여부 (1=yes, 0=no)
    # - make_sorted: 출력을 정렬할지 (1=yes, 0=no)
    # - keep_duplicates: 중복 허용 여부 (1=yes, 0=no)
    # - num_truncation_strategies: truncation 전략 개수
    # - truncation_strategies: truncation 전략 배열
    # - workspace: workspace descriptor
    cupp.pauli_expansion_view_compute_operator_application(
        handle,
        input_view,
        current_output,
        gate,
        1,  # adjoint=True
        0,  # don't sort
        0,  # no duplicates
        0,  # no truncation
        None,
        workspace
    )
    
    # View 삭제
    cupp.destroy_pauli_expansion_view(input_view)
    
    # Input/output swap
    current_input, current_output = current_output, current_input
    
    # 진행 상황 출력
    if i % 2 == 0 or i == 0:
        new_num_terms = cupp.pauli_expansion_get_num_terms(handle, current_input)
        print(f"   After gate {len(circuit) - i}: {new_num_terms} Pauli terms")

print("\n   ✓ Pauli propagation completed")

# ============================================================================
# 8. 기댓값 계산: Tr(|0⟩⟨0| * evolved_observable)
# ============================================================================
print(f"\n8. Computing expectation value...")

# 최종 expansion의 term 개수
final_num_terms = cupp.pauli_expansion_get_num_terms(handle, current_input)
print(f"   Final number of Pauli terms: {final_num_terms}")

# 최종 view 생성
final_view = cupp.pauli_expansion_get_contiguous_range(
    handle, current_input, 0, final_num_terms
)

# pauli_expansion_view_compute_trace_with_zero_state() 파라미터:
# - handle: cuPauliProp handle
# - view: Pauli expansion view
# - result: 결과를 저장할 포인터 (numpy array)
# - workspace: workspace descriptor
# 
# 계산: Tr(view * |0⟩⟨0|)
# = Tr(evolved_observable * |0⟩⟨0|)
# = ⟨0|evolved_observable|0⟩
result = np.array([0.0], dtype=np.float64)
cupp.pauli_expansion_view_compute_trace_with_zero_state(
    handle,
    final_view,
    result.ctypes.data,
    workspace
)

expectation_value = result[0]

print(f"\n{'=' * 60}")
print(f"RESULT: ⟨ψ|Z_0|ψ⟩ = {expectation_value:.6f}")
print(f"{'=' * 60}")

# ============================================================================
# 9. Cleanup
# ============================================================================
print(f"\n9. Cleaning up resources...")

cupp.destroy_pauli_expansion_view(final_view)

for gate in circuit:
    cupp.destroy_operator(gate)

cupp.destroy_pauli_expansion(input_expansion)
cupp.destroy_pauli_expansion(output_expansion)
cupp.destroy_workspace_descriptor(workspace)
cupp.destroy(handle)

print("   ✓ All resources freed")
print("\n✓ Simulation completed successfully!")
print()
