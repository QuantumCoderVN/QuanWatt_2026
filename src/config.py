# ============================================================
# config.py
# ============================================================

import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FDLS
# ============================================================

FDLS_MAX_ITER = 8
FDLS_TOL = 1e-7


# ============================================================
# HHL CONFIG
# ============================================================

HHL_PHASE_QUBITS = 8

# Không dùng trị riêng thật.
# Dùng bound để chọn t_eff sao cho phase lớn nhất khoảng 0.40.
HHL_PHASE_TARGET = 0.40

HHL_C = 1.0

# Chỉ bật khi debug, không dùng trong bản chính.
HHL_DEBUG_COMPARE_CLASSICAL = False


# ============================================================
# VQLS CONFIG
# ============================================================

VQLS_STEPS = 100
VQLS_RHOBEG = 0.5
VQLS_Q_DELTA = 0.001
VQLS_RNG_SEED = 0

PAULI_ATOL = 1e-10
PAULI_RTOL = 1e-10
MAX_PAULI_TERMS = None

USE_COMPLEX_ANSATZ = True
VQLS_LAYERS = 2


# ============================================================
# PRINT CONFIG
# ============================================================

# False: không in chi tiết từng vòng FDLS
PRINT_EVERY_ITER = False

# False: không in chi tiết trong Classical/HHL/VQLS mỗi lần solve
PRINT_SOLVER_DETAIL = False