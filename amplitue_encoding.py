"""
Amplitude encoding cho vector b bất kỳ (thực hoặc phức):
    |0...0>  -->  |b> = sum_i (b_i / ||b||) |i>

Thuật toán: Möttönen et al. (2004) "Transformation of quantum states using
uniformly controlled rotations" + Shende-Bullock-Markov (2006).

Ý tưởng:
  1. Đệm 0 để độ dài N = 2^n, chuẩn hoá b.
  2. Xem |b| như lá của một cây nhị phân độ sâu n. Ở tầng k (qubit k),
     mỗi nút (tiền tố j của k bit đầu) chia "khối lượng xác suất" sang con
     trái/phải bằng một phép quay RY(theta), theta = 2*atan2(||phải||, ||trái||).
     Phép quay này phải được điều khiển bởi giá trị j của k qubit trước
     -> "uniformly controlled RY" (UC-RY).
  3. Pha: với cặp con có pha (phi0, phi1), dùng RZ(phi1 - phi0) và đẩy pha
     trung bình (phi0+phi1)/2 lên nút cha -> UC-RZ ở mỗi tầng.
  4. Mỗi UC-R với k qubit điều khiển được phân rã thành đúng 2^k phép quay
     đơn qubit + 2^k CNOT (mã Gray), với góc mới
         theta'_i = 2^-k * sum_j (-1)^{popcount(j & gray(i))} alpha_j
  Tổng: ~ 2^(n+1) CNOT, độ sâu O(2^n) – tối ưu tiệm cận cho trạng thái tổng quát.

Quy ước: qubit 0 là bit cao nhất (MSB) của chỉ số i (giống sách giáo khoa;
Qiskit dùng little-endian nên khi xuất sang Qiskit ta đảo thứ tự qubit).

Chỉ cần numpy. Qiskit là tuỳ chọn (hàm to_qiskit).
"""
import numpy as np

# ----------------------------------------------------------------------------
# 1. Tính góc trên cây nhị phân
# ----------------------------------------------------------------------------

def prepare_vector(b, tol=1e-12):
    b = np.asarray(b, dtype=complex).ravel()
    if b.size == 0 or np.linalg.norm(b) < tol:
        raise ValueError("Vector phải khác 0.")
    n = max(1, int(np.ceil(np.log2(b.size))))
    psi = np.zeros(2 ** n, dtype=complex)
    psi[: b.size] = b
    return psi / np.linalg.norm(psi), n


def compute_angles(psi, n):
    """Trả về (ry_angles, rz_angles, global_phase).
    ry_angles[k], rz_angles[k] là mảng 2^k góc cho tầng k (qubit k)."""
    mags = np.abs(psi)
    phases = np.angle(psi)
    ry, rz = [None] * n, [None] * n

    # Đi từ lá lên gốc
    level_mag, level_phase = mags, phases
    for k in range(n - 1, -1, -1):
        left_m, right_m = level_mag[0::2], level_mag[1::2]
        left_p, right_p = level_phase[0::2], level_phase[1::2]
        ry[k] = 2 * np.arctan2(right_m, left_m)
        rz[k] = right_p - left_p
        level_mag = np.sqrt(left_m ** 2 + right_m ** 2)
        level_phase = (left_p + right_p) / 2
    global_phase = level_phase[0]
    return ry, rz, global_phase


# ----------------------------------------------------------------------------
# 2. Phân rã phép quay điều khiển đồng nhất (UC-R) thành CNOT + R
# ----------------------------------------------------------------------------

def _gray(i):
    return i ^ (i >> 1)


def uniformly_controlled_rotation(axis, alphas, controls, target, tol=1e-12):
    """Trả về list cổng [(tên, tham số, qubits)].
    alphas[j] = góc khi các qubit điều khiển (controls[0] là MSB) có giá trị j."""
    alphas = np.asarray(alphas, dtype=float)
    k = len(controls)
    if k == 0:
        return [] if abs(alphas[0]) < tol else [(axis, alphas[0], (target,))]
    if np.allclose(alphas, 0, atol=tol):
        return []

    size = 2 ** k
    # theta'_i = 2^-k * sum_j (-1)^{popcount(j & gray(i))} alpha_j
    M = np.array([[(-1) ** bin(j & _gray(i)).count("1") for j in range(size)]
                  for i in range(size)])
    thetas = M @ alphas / size

    gates = []
    for i in range(size):
        if abs(thetas[i]) > tol:
            gates.append((axis, thetas[i], (target,)))
        # bit thay đổi giữa gray(i) và gray(i+1) (vòng về 0 ở cuối)
        diff = _gray(i) ^ _gray((i + 1) % size)
        bit = diff.bit_length() - 1          # vị trí bit (LSB = 0)
        ctrl = controls[k - 1 - bit]         # controls[0] là MSB
        gates.append(("cx", None, (ctrl, target)))
    return gates


# ----------------------------------------------------------------------------
# 3. Dựng mạch hoàn chỉnh
# ----------------------------------------------------------------------------

def amplitude_encoding_circuit(b):
    """Trả về (gates, n_qubits, global_phase)."""
    psi, n = prepare_vector(b)
    ry, rz, gphase = compute_angles(psi, n)
    gates = []
    for k in range(n):
        controls = list(range(k))
        gates += uniformly_controlled_rotation("ry", ry[k], controls, k)
        gates += uniformly_controlled_rotation("rz", rz[k], controls, k)
    return gates, n, gphase


# ----------------------------------------------------------------------------
# 4. Bộ mô phỏng statevector nhỏ để kiểm tra
# ----------------------------------------------------------------------------

def _apply_1q(state, U, q, n):
    state = state.reshape([2] * n)
    state = np.moveaxis(state, q, 0)
    state = np.tensordot(U, state, axes=([1], [0]))
    return np.moveaxis(state, 0, q).reshape(-1)


def _apply_cx(state, c, t, n):
    state = state.reshape([2] * n).copy()
    idx1 = [slice(None)] * n
    idx1[c] = 1
    sub = state[tuple(idx1)]
    t_ax = t if t < c else t - 1
    state[tuple(idx1)] = np.flip(sub, axis=t_ax)
    return state.reshape(-1)


def simulate(gates, n, global_phase=0.0):
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1
    for name, theta, qs in gates:
        if name == "ry":
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            state = _apply_1q(state, np.array([[c, -s], [s, c]]), qs[0], n)
        elif name == "rz":
            U = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
            state = _apply_1q(state, U, qs[0], n)
        elif name == "cx":
            state = _apply_cx(state, qs[0], qs[1], n)
    return state * np.exp(1j * global_phase)


# ----------------------------------------------------------------------------
# 5. (Tuỳ chọn) Xuất sang Qiskit
# ----------------------------------------------------------------------------

def to_qiskit(b):
    from qiskit import QuantumCircuit
    gates, n, gphase = amplitude_encoding_circuit(b)
    qc = QuantumCircuit(n, global_phase=gphase)
    q = lambda i: n - 1 - i          # đổi MSB-first -> little-endian của Qiskit
    for name, theta, qs in gates:
        if name == "ry":
            qc.ry(theta, q(qs[0]))
        elif name == "rz":
            qc.rz(theta, q(qs[0]))
        else:
            qc.cx(q(qs[0]), q(qs[1]))
    return qc


def print_circuit(gates):
    for name, theta, qs in gates:
        if name == "cx":
            print(f"CX   q{qs[0]} -> q{qs[1]}")
        else:
            print(f"{name.upper()}({theta:+.4f})  q{qs[0]}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Ví dụ 1: vector thực nhỏ
    b = [1, 2, 3, 4]
    gates, n, gp = amplitude_encoding_circuit(b)
    print(f"b = {b}  ->  {n} qubit, {sum(g[0]=='cx' for g in gates)} CNOT")
    print_circuit(gates)