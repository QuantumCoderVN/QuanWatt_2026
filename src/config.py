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
# HHL - theo code HHL của bạn
# ============================================================

# Số phase qubit cho QPE
HHL_PHASE_QUBITS = 8

# Trong code HHL của bạn có:
# t = 0.2
#
# Nhưng với B' từ IEEE, trị riêng thường lớn hơn ví dụ toy của bạn.
# Do đó để tránh phase bị wrap, dùng t nhỏ hơn.
# Bản chất vẫn là đúng code HHL:
# U = exp(i A t)
HHL_T = 0.12

# Trong code HHL của bạn:
# control_rotation_gate(..., C=1.0)
HHL_C = 1.0


# ============================================================
# VQLS - theo code VQLS của bạn
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
# PRINT
# ============================================================

PRINT_EVERY_ITER = True