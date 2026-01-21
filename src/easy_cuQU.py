from cuquantum.bindings import cupauliprop as cupp
import re
import cupy as cp
import time

__version__ = "0.1.0"

def observable_evolution(
    handle=None, gate_ops=None, input_expansion=None, output_expansion=None, workspace=None,
    truncation_strategies=None, ctx=None,
    d_input_pauli_buffer=None, d_input_coef_buffer=None, d_output_pauli_buffer=None, d_output_coef_buffer=None
):
    """
    Pauli observable propagation through a circuit.
    Returns: (evolved_expansion, final_num_terms, input_expansion, output_expansion)
    """
    # Robustly set num_qubits for debug print
    num_qubits = None
    if ctx is not None:
        handle = ctx.handle
        num_qubits = ctx.num_qubits
        gate_ops = ctx.gate_ops
        workspace = ctx.workspace
        max_terms = ctx.max_terms
        num_packed_ints = ctx.num_packed_ints
        d_input_pauli_buffer = ctx.d_input_pauli_buffer
        d_input_coef_buffer = ctx.d_input_coef_buffer
        d_output_pauli_buffer = ctx.d_output_pauli_buffer
        d_output_coef_buffer = ctx.d_output_coef_buffer

        pauli_buffer_size = d_input_pauli_buffer.nbytes
        coef_buffer_size = d_input_coef_buffer.nbytes
        input_expansion = cupp.create_pauli_expansion(
            handle, num_qubits,
            d_input_pauli_buffer.data.ptr, pauli_buffer_size,
            d_input_coef_buffer.data.ptr, coef_buffer_size,
            1,  # dataType: CUDA_R_64F
            int(cp.count_nonzero(d_input_coef_buffer)),  # numTerms
            1,  # isSorted
            1   # isUnique
        )
        pauli_buffer_size_out = d_output_pauli_buffer.nbytes
        coef_buffer_size_out = d_output_coef_buffer.nbytes
        output_expansion = cupp.create_pauli_expansion(
            handle, num_qubits,
            d_output_pauli_buffer.data.ptr, pauli_buffer_size_out,
            d_output_coef_buffer.data.ptr, coef_buffer_size_out,
            1, 0, 0, 0  # empty
        )
    else:
        if handle is None or gate_ops is None or input_expansion is None or output_expansion is None or workspace is None:
            raise ValueError("observable_evolution: ctx가 없으면 handle, gate_ops, input_expansion, output_expansion, workspace를 모두 인자로 넘겨야 합니다.")
        if 'max_terms' in locals():
            max_terms = max_terms
        else:
            max_terms = int(1e4)
        if 'num_packed_ints' in locals():
            num_packed_ints = num_packed_ints
        else:
            num_packed_ints = 1
        d_input_pauli_buffer = locals().get('d_input_pauli_buffer', None)
        d_input_coef_buffer = locals().get('d_input_coef_buffer', None)
        d_output_pauli_buffer = locals().get('d_output_pauli_buffer', None)
        d_output_coef_buffer = locals().get('d_output_coef_buffer', None)
        if d_input_pauli_buffer is None or d_input_coef_buffer is None or d_output_pauli_buffer is None or d_output_coef_buffer is None:
            raise ValueError("observable_evolution: ctx 없이 호출 시 *_buffer도 인자로 넘겨야 합니다.")
        # Try to infer num_qubits from input_expansion if not set
        if num_qubits is None:
            try:
                num_qubits = input_expansion.num_qubits
            except Exception:
                num_qubits = 0

    current_input = input_expansion
    current_output = output_expansion
    s = time.time()

    for gate_idx, gate in enumerate(reversed(gate_ops)):
        gate_number = len(gate_ops) - gate_idx
        num_terms = cupp.pauli_expansion_get_num_terms(handle, current_input)
        print(f"Gate {gate_number} (reverse): {num_terms} input terms")

        input_view = cupp.pauli_expansion_get_contiguous_range(
            handle, current_input, 0, num_terms
        )

        cupp.pauli_expansion_view_compute_operator_application(
            handle, input_view, current_output, gate,
            1,  # adjoint
            1,  # sort
            0,  # keep_duplicates
            0,  # num_truncation_strategies
            truncation_strategies,
            workspace,
        )

        cupp.destroy_pauli_expansion_view(input_view)
        current_input, current_output = current_output, current_input

        new_num_terms = cupp.pauli_expansion_get_num_terms(handle, current_input)
        print(f"  → {new_num_terms} output terms")
        if new_num_terms >= max_terms:
            raise RuntimeError(
                f"Pauli expansion term overflow: {new_num_terms} >= max_terms={max_terms}.")

        # 디버깅 출력 (읽기 전용)
        if current_input == input_expansion:
            buffer_pauli = d_input_pauli_buffer
            buffer_coef = d_input_coef_buffer
        else:
            buffer_pauli = d_output_pauli_buffer
            buffer_coef = d_output_coef_buffer
        pauli_data = buffer_pauli[:new_num_terms*2*num_packed_ints].get()
        coef_data = buffer_coef[:new_num_terms].get()
        for j in range(min(new_num_terms, 2)):
            x_mask = pauli_data[2*j]; z_mask = pauli_data[2*j + 1]; coef = coef_data[j]
            if num_qubits is not None and num_qubits > 0:
                print(f"    Term {j}: X=0b{x_mask:0{num_qubits}b} Z=0b{z_mask:0{num_qubits}b} coef={coef:+.4f}")
            else:
                print(f"    Term {j}: X=0b{x_mask:b} Z=0b{z_mask:b} coef={coef:+.4f}")
        if new_num_terms > 5:
            print(f"    ... and {new_num_terms - 5} more terms")
        print()

    final_num_terms = cupp.pauli_expansion_get_num_terms(handle, current_input)
    print(f"✓ Propagation 완료: 최종 {final_num_terms}개 Pauli terms")
    print(f"총 소요 시간: {time.time() - s:.2f} 초")
    # If ctx is provided, store results for downstream use
    if ctx is not None:
        ctx.evolved_expansion = current_input
        ctx.final_num_terms = final_num_terms
        ctx.input_expansion = input_expansion
        ctx.output_expansion = output_expansion
    return current_input, final_num_terms, input_expansion, output_expansion


def compute_expectation(
    handle=None, evolved_expansion=None, final_num_terms=None, input_expansion=None, output_expansion=None,
    ctx=None,
    num_packed_ints=None, num_qubits=None, workspace=None,
    d_input_pauli_buffer=None, d_input_coef_buffer=None, d_output_pauli_buffer=None, d_output_coef_buffer=None
):
    """
    Calculate expectation value ⟨0|O'|0⟩ for the evolved observable.
    ctx: must contain *_buffer, num_packed_ints, workspace
    """
    # ctx가 있으면 ctx에서 모든 값 추출
    if ctx is not None:
        if handle is None:
            handle = ctx.handle
        if evolved_expansion is None:
            evolved_expansion = getattr(ctx, 'evolved_expansion', None)
        if final_num_terms is None:
            final_num_terms = getattr(ctx, 'final_num_terms', None)
        if input_expansion is None:
            input_expansion = getattr(ctx, 'input_expansion', None)
        if output_expansion is None:
            output_expansion = getattr(ctx, 'output_expansion', None)
        num_packed_ints = ctx.num_packed_ints
        num_qubits = ctx.num_qubits
        workspace = ctx.workspace
        d_input_pauli_buffer = ctx.d_input_pauli_buffer
        d_input_coef_buffer = ctx.d_input_coef_buffer
        d_output_pauli_buffer = ctx.d_output_pauli_buffer
        d_output_coef_buffer = ctx.d_output_coef_buffer
        # evolved_expansion, final_num_terms 등은 observable_evolution 결과를 ctx에 저장해두면 자동으로 사용됨
        if evolved_expansion is None or final_num_terms is None or input_expansion is None or output_expansion is None:
            raise ValueError("compute_expectation: ctx 사용 시 evolved_expansion, final_num_terms, input_expansion, output_expansion이 ctx에 포함되어야 합니다.")
        if evolved_expansion == input_expansion:
            buffer_pauli = d_input_pauli_buffer
            buffer_coef = d_input_coef_buffer
        else:
            buffer_pauli = d_output_pauli_buffer
            buffer_coef = d_output_coef_buffer
    else:
        # ctx 없이 수동 인자 사용
        # 필수 인자: num_packed_ints, num_qubits, workspace, 버퍼들
        if handle is None or evolved_expansion is None or final_num_terms is None or input_expansion is None or output_expansion is None or \
           num_packed_ints is None or num_qubits is None or workspace is None or \
           d_input_pauli_buffer is None or d_input_coef_buffer is None or \
           d_output_pauli_buffer is None or d_output_coef_buffer is None:
            raise ValueError("compute_expectation: ctx 없이 호출 시 handle, evolved_expansion, final_num_terms, input_expansion, output_expansion, num_packed_ints, num_qubits, workspace, *_buffer 모두 인자로 넘겨야 합니다.")
        if evolved_expansion == input_expansion:
            buffer_pauli = d_input_pauli_buffer
            buffer_coef = d_input_coef_buffer
        else:
            buffer_pauli = d_output_pauli_buffer
            buffer_coef = d_output_coef_buffer
        if evolved_expansion == input_expansion:
            buffer_pauli = d_input_pauli_buffer
            buffer_coef = d_input_coef_buffer
        else:
            buffer_pauli = d_output_pauli_buffer
            buffer_coef = d_output_coef_buffer

    pauli_data = buffer_pauli[:final_num_terms*2*num_packed_ints].get()
    coef_data = buffer_coef[:final_num_terms].get()
    print("최종 Pauli Expansion 상태:")
    for j in range(final_num_terms):
        x_mask = pauli_data[2*j]
        z_mask = pauli_data[2*j + 1]
        coef = coef_data[j]
        print(f"  Term {j}: X={x_mask:0{num_qubits}b} Z={z_mask:0{num_qubits}b} coef={coef:.6f}")
    print()

    final_view = cupp.pauli_expansion_get_contiguous_range(
        handle, evolved_expansion, 0, final_num_terms
    )
    import numpy as np
    result = np.array([0.0], dtype=np.float64)
    cupp.pauli_expansion_view_compute_trace_with_zero_state(
        handle,
        final_view,
        result.ctypes.data,
        workspace
    )
    print("=" * 60)
    print(f"결과: ⟨ψ|O|ψ⟩ = {result[0]:.6f}")
    print("=" * 60)
    return result[0]


def create_observable(terms: list[tuple[float, str]], num_qubits: int = None) -> list[tuple[float, int, int]]:
    """ 사용자 친화적 형식 → cuPauliProp 형식 변환     
    Args:
        terms: List of (coefficient, pauli_string) tuples
               예: [(0.8, "X0 X2 Z1"), (-0.6, "X4 Z3")]
        num_qubits: 큐비트 개수 (범위 검증용, None이면 검증 안 함)
    Returns:
        List of (coef, X_mask, Z_mask) tuples
    """

    def pauli_string_to_masks(pauli_str):
        """ Pauli 문자열을 비트마스크로 변환
        Examples:
            "X0 X2 Z1" → X on qubits 0,2 and Z on qubit 1
            "Z10"      → Z on qubit 10
            ""         → Identity (no operations)
        Returns:
            (X_mask, Z_mask) as integers
        """
        X_mask = 0;     Z_mask = 0
        
        # "X0", "Y2", "Z10" 형태 파싱
        pattern = r'([XYZI])(\d+)'
        matches = re.findall(pattern, pauli_str.upper())
        
        for pauli, qubit_str in matches:
            q = int(qubit_str)
            
            # 범위 검증
            if num_qubits is not None and q >= num_qubits:
                raise ValueError(
                    f"❌ Qubit index {q} out of range! "
                    f"시스템 큐비트 수: {num_qubits} (유효 범위: 0~{num_qubits-1})\n"
                    f"문제 항: '{pauli_str}'"
                    f""
                )
            
            # Left Shift 연산
            if pauli == 'X':
                X_mask |= (1 << q)  # 2**q와 동일
            elif pauli == 'Z':
                Z_mask |= (1 << q)
            elif pauli == 'Y':
                X_mask |= (1 << q)
                Z_mask |= (1 << q)
            # 'I'는 무시 (identity)
        
        return X_mask, Z_mask


    obs_terms = []
    for coef, pauli_str in terms:
        X_mask, Z_mask = pauli_string_to_masks(pauli_str)
        obs_terms.append((coef, X_mask, Z_mask))

    print('당신이 정의한 관측자의 mask 표현:')
    for i, (coef, xm, zm) in enumerate(obs_terms):
        print(f"Term {i}: coef={coef:+.1f}, X_mask=0b{xm:010b}, Z_mask=0b{zm:010b}")
    return obs_terms

# 사용 예시

def observable_to_cuQU_input(obs_terms: list[tuple[float, int, int]]) -> tuple[cp.ndarray, cp.ndarray]:
    """ cuPauliProp 입력 형식으로 변환
    Args:
        obs_terms: List of (coef, X_mask, Z_mask) tuples
    Returns:
        Two lists: pauli_list and coef_list
    """
    print('당신의 관측자가 cuPauliProp 입력 형식으로 변환됩니다')
    pauli_list = []
    coef_list = []
    for coef, X_mask, Z_mask in obs_terms:
        pauli_list.extend([X_mask, Z_mask])
        coef_list.append(coef)

    
    d_input_pauli = cp.array(pauli_list, dtype=cp.uint64)
    d_input_coef  = cp.array(coef_list, dtype=cp.float64)
    return d_input_pauli, d_input_coef


def get_packed_ints(num_qubits: int) -> int:
    """ 주어진 큐비트 수에 필요한 packed integers 수 계산
    Args:
        num_qubits: Number of qubits
    Returns:
        Number of packed integers needed
    """
    num_packed_ints = (num_qubits + 63) // 64
    print(f"큐비트 수: {num_qubits}, 필요한 packed integers 수: {num_packed_ints}")
    return num_packed_ints


def make_handle() -> int:
    """
    Make cuPauliProp handle and print initialization message.
    Returns: int == cuPauliProp handle == ptr to internal cuPauliProp structure
    """    
    
        # 새로운 handle 생성
    handle = cupp.create()
    print("cuPauliProp 핸들 생성 완료.")
    return handle


def cleanup_cupauli():
    """
    cuPauliProp 리소스 최종 정리
    
    사용법: 노트북 맨 마지막에서 호출
        cleanup_cupauli()
    """
    print("리소스 정리 중...")
    
    # Gate operators 삭제
    try:
        if 'gate_ops' in globals():
            for gate in gate_ops:
                try:
                    cupp.destroy_operator(gate)
                    print(f"  ✓ {len(gate_ops)}개 gate operators 삭제")
                except:
                    pass
    except Exception as e:
        print(f"  Gate operators 정리 중 에러 (무시): {e}")
    
    # Expansions 삭제
    try:
        if 'input_expansion' in globals():
            cupp.destroy_pauli_expansion(input_expansion)
            print("  ✓ Pauli input expansions 삭제")
        if 'output_expansion' in globals():
            cupp.destroy_pauli_expansion(output_expansion)
            print("  ✓ Pauli output expansions 삭제")
    except Exception as e:
        print(f"  Expansions 정리 중 에러 (무시): {e}")
    
    # Workspace 삭제
    try:
        if 'workspace' in globals():
            cupp.destroy_workspace_descriptor(workspace)
            print("  ✓ Workspace 삭제")        
    except Exception as e:
        print(f"  Workspace 정리 중 에러 (무시): {e}")
    
    # Handle 삭제
    try:
        if 'handle' in globals():
            cupp.destroy(handle)
            print("  ✓ Handle 삭제")
    except Exception as e:
        print(f"  Handle 정리 중 에러 (무시): {e}")
    
    print("✓ 모든 리소스 해제 완료. 새로운 시작을 위해 준비되었습니다.")


def create_buffer(NUM_QUBITS=None, d_input_pauli=None, d_input_coef=None, max_terms=None,ctx = None) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:    
    if ctx:
        NUM_QUBITS = ctx.num_qubits
        d_input_pauli = ctx.d_input_pauli
        d_input_coef = ctx.d_input_coef

    num_packed_ints = get_packed_ints(NUM_QUBITS)
    if max_terms == None:
        max_terms = int(1e+4)
        print(f"max terms의 기본 값은 {max_terms}개 입니다.")
    elif max_terms > int(1e+7):        
        print(f"경고: max_terms의 최대값 {1e+7}을 초과합니다. {1e+7}로 설정됩니다.")
        max_terms = 1e+7
    elif max_terms < 1e+4:
        max_terms = int(1e+4)
        print(f"max_terms가 최소 {1e+4}개입니다. {max_terms}로 설정합니다.")
    max_terms = int(max_terms)
    print(f"max_terms 설정값: {max_terms}")


    # GPU 메모리에 저장공간 확보 및 초기 데이터 복사
    ########################################################################################
    

    pauli_buffer_size = 2 * num_packed_ints * max_terms * 8  # bytes
    coef_buffer_size = max_terms * 8  # bytes (float64)

    # Input/Output buffers
    d_input_pauli_buffer = cp.zeros(pauli_buffer_size, dtype=cp.uint64)
    d_output_pauli_buffer = cp.zeros(pauli_buffer_size, dtype=cp.uint64)
    
    d_input_coef_buffer = cp.zeros(coef_buffer_size, dtype=cp.float64)
    d_output_coef_buffer = cp.zeros(coef_buffer_size, dtype=cp.float64)
    
    # 버퍼 초기화
    d_input_pauli_buffer.fill(0)
    d_input_coef_buffer.fill(0)
    d_output_pauli_buffer.fill(0)
    d_output_coef_buffer.fill(0)

    # 초기 데이터 복사
    d_input_pauli_buffer[:len(d_input_pauli)] = d_input_pauli  
    d_input_coef_buffer[:len(d_input_coef)] = d_input_coef
    ##########################################################################################

    return max_terms, (d_input_pauli_buffer, d_input_coef_buffer,
            d_output_pauli_buffer, d_output_coef_buffer)


# 공용 정리 함수: workspace 해제 + CuPy 메모리 캐시 반환
def cleanup_workspace_and_memory():
    global workspace
    ws_prev = globals().get('workspace', None)
    if ws_prev is not None:
        try:
            cupp.destroy_workspace_descriptor(ws_prev)
            print("✓ 이전 workspace 해제됨")
        except Exception:
            pass
        workspace = None

    # scratch 포인터 참조 제거 (풀 캐시 반환을 유도)
    globals()['workspace_mem'] = None

    # CuPy 메모리 풀 비우기 (사용 중이지 않은 캐시 블록 반환)
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Device().synchronize()
        free_mem, total_mem = cp.cuda.Device().mem_info
        print(f"✓ 메모리 풀 비움: 현재 가용 {free_mem/1024**3:.2f} GB / 총 {total_mem/1024**3:.2f} GB")
    except Exception as e:
        print(f"메모리 풀 비우기 중 경고: {e}")


def create_workspace(handle=None, d_input_pauli_buffer=None, d_input_coef_buffer=None, d_output_pauli_buffer=None, d_output_coef_buffer=None, ctx=None, frac = 0.8):
    if ctx:
        handle = ctx.handle
        d_input_pauli_buffer = ctx.d_input_pauli_buffer
        d_input_coef_buffer = ctx.d_input_coef_buffer
        d_output_pauli_buffer = ctx.d_output_pauli_buffer
        d_output_coef_buffer = ctx.d_output_coef_buffer
    
    # 전역 workspace 사용
    global workspace

    # 먼저 기존 리소스/캐시를 정리해 가용 메모리를 확보
    cleanup_workspace_and_memory()

    # 가용 메모리 확인
    free_mem, _ = cp.cuda.Device().mem_info

    # Workspace 크기 계산
    buffer_mem = (
        d_input_pauli_buffer.nbytes + d_input_coef_buffer.nbytes +
        d_output_pauli_buffer.nbytes + d_output_coef_buffer.nbytes
    )
    workspace_size = int(max(0, (free_mem - buffer_mem)) * frac)
    if workspace_size <= 0:
        raise RuntimeError("workspace_size <= 0: 버퍼/가용 메모리 확인 필요")

    # 새로 할당
    print("workspace를 할당합니다.")
    print(f"가용 메모리 {free_mem / 1024**3:.2f} GB의 {int(frac*100)}%인 {workspace_size / 1024**3:.2f} GB를 사용합니다")

    # scratch 메모리 포인터를 전역에 저장해 생존 보장
    d_workspace = cp.cuda.alloc(workspace_size)
    globals()['workspace_mem'] = d_workspace

    # workspace descriptor 생성 및 메모리 설정
    ws_desc = cupp.create_workspace_descriptor(handle)
    cupp.workspace_set_memory(
        handle, ws_desc,
        0,  # CUPAULIPROP_MEMSPACE_DEVICE
        0,  # CUPAULIPROP_WORKSPACE_SCRATCH
        d_workspace.ptr, workspace_size
    )

    # 전역 workspace 핸들 저장 및 반환
    workspace = ws_desc
    return ws_desc


def reset_expansions(handle=None, NUM_QUBITS=None, d_input_pauli_buffer=None, d_input_coef_buffer=None, d_output_pauli_buffer=None, d_output_coef_buffer=None, ctx = None):
    if ctx:
        handle = ctx.handle
        NUM_QUBITS = ctx.num_qubits
        d_input_pauli_buffer = ctx.d_input_pauli_buffer
        d_input_coef_buffer = ctx.d_input_coef_buffer
        d_output_pauli_buffer = ctx.d_output_pauli_buffer
        d_output_coef_buffer = ctx.d_output_coef_buffer

    try:
        if 'input_expansion' in globals():
            cupp.destroy_pauli_expansion(input_expansion)
        if 'output_expansion' in globals():
            cupp.destroy_pauli_expansion(output_expansion)
    except Exception:
        pass

    input_expansion = cupp.create_pauli_expansion(
        handle,      NUM_QUBITS,
        d_input_pauli_buffer.data.ptr,  len(d_input_pauli_buffer), # pauli_buffer_size == len(d_input_pauli_buffer)
        d_input_coef_buffer.data.ptr,   len(d_input_coef_buffer),          # coef_buffer_size == len(d_input_coef_buffer)
        1,                           # data_type -> 1은 float64 (double)을 의미
        int(cp.argmax(d_input_coef_buffer==0)),           # number of terms == d_input_coef_buffer에서 처음으로 0이 나오는 위치 == 그전까지 계수가 있음 == term의 개수   
        0,                           # 정렬되어 있는가?  아마 처음에는 무작위 순서겠지
        1,                           # 중복이 있는가? (초기에는 안전을 위해 있을수도 있다고 두자)
    )

    output_expansion = cupp.create_pauli_expansion(
        handle,       NUM_QUBITS,
        d_output_pauli_buffer.data.ptr,  len(d_output_pauli_buffer), # pauli_buffer_size == len(d_output_pauli_buffer)
        d_output_coef_buffer.data.ptr,   len(d_output_coef_buffer),  # coef_buffer_size == len(d_output_coef_buffer)
        1,                           # data_type -> 1은 float64 (double)을 의미
        0,                           # 정렬되어 있는가?  아마 처음에는 무작위 순서겠지
        0,
        0,
    )

    return input_expansion, output_expansion


class CircuitBuilder:
    def __init__(self, handle, num_qubits):
        self.pauli_dict = {"I":0, "X":1, "Y":2, "Z":3}
        self.clifford_dict = {"I":0, "X":1, "Y":2, "Z":3, "H":4, "S":5,
                              "CNOT":7, "CZ":8, "CY":9, "SWAP":10, "ISWAP":11,
                              "SQRTX":12, "SQRTZ":13, "SQRTY":14}        
        self.handle = handle
        self.num_qubits = num_qubits
        self.ops = []

    ###################### ROTATION GATES ######################
    def rx(self, qubit, angle):
        self.ops.append(cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, 1, [qubit], [1]))
        return self  # 체이닝용
    
    def ry(self, qubit, angle):
        self.ops.append(cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, 1, [qubit], [3]))
        return self
    
    def rz(self, qubit, angle):
        self.ops.append(cupp.create_pauli_rotation_gate_operator(
            self.handle, angle, 1, [qubit], [2]))
        return self
    
    ################## CLIFFORD GATES ##################
    def x(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 1, [qubit]))
        return self
        
    def y(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 3, [qubit]))
        return self

    def z(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 2, [qubit]))
        return self

    def h(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 4, [qubit]))
        return self
    
    def s(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 5, [qubit]))
        return self
    
    def cnot(self, control, target):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 7, [target, control]))
        return self

    def cy(self, control, target):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 9, [target, control]))
        return self
    
    def cz(self, control, target):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 8, [target, control]))
        return self

    def swap(self, qubit1, qubit2):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 10, [qubit1, qubit2]))
        return self

    def iswap(self, qubit1, qubit2):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 11, [qubit1, qubit2]))
        return self 
    
    def sqrtx(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 12, [qubit]))
        return self
    
    def sqrty(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 14, [qubit]))
        return self
    
    def sqrtz(self, qubit):
        self.ops.append(cupp.create_clifford_gate_operator(
            self.handle, 13, [qubit]))
        return self
      
    ###################### BUILD ######################
    def build(self) -> list: # 게이트 오퍼레이션을 담은 리스트 반환
        print(f"✓ {len(self.ops)}개 gate operators 생성 완료")
        return self.ops
