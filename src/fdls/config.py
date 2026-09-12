# ============================================================
# config.py
# ============================================================

import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FDLS
# ============================================================

# Kept identical to the original configuration.
FDLS_MAX_ITER = 8
FDLS_TOL = 1e-7


# ============================================================
# HHL CONFIG
# ============================================================

HHL_PHASE_QUBITS = 8

# Do not use exact eigenvalues. Use a bound so the largest phase is ~0.40.
HHL_PHASE_TARGET = 0.40

HHL_C = 1.0

# Enable only for debugging.
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

PRINT_EVERY_ITER = False
PRINT_SOLVER_DETAIL = False