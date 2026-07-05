# ============================================================
# config_qopf3.py
# Config for IEEE14-subcase-3bus DC-OPF / SC-PDIPM / VQLS
# ============================================================

# ============================================================
# Case selection
# ============================================================

SELECTED_BUSES = (1, 2, 3)


# ============================================================
# Printing
# ============================================================

PRINT_MATRIX_DETAIL = True
PRINT_PDIPM_ITER = True


# ============================================================
# Physical constraints
# ============================================================

ENFORCE_GEN_BOUNDS = True

ENFORCE_THETA_BOUNDS = True
THETA_MIN_DEG = -60.0
THETA_MAX_DEG = 60.0

ENFORCE_BRANCH_LIMITS = True

ACTIVE_TOL = 1e-7


# ============================================================
# SC-PDIPM configuration
# ============================================================

PDIPM_MAX_ITERS = 80

PDIPM_TOL_FEAS = 1e-9
PDIPM_TOL_GRAD = 1e-9
PDIPM_TOL_COMP = 1e-9

PDIPM_XI = 0.995
PDIPM_SIGMA = 0.10
PDIPM_GAMMA0 = 1.0
PDIPM_MU0 = 1.0

PDIPM_USE_STEP_CONTROL = True
PDIPM_STEP_KAPPA = 0.5
PDIPM_MAX_LINESEARCH = 20


# ============================================================
# Linear solver inside SC-PDIPM
# ============================================================
#
# classical:
#     dùng np.linalg.solve để update, ổn định nhất.
#
# compare:
#     dùng classical để update, nhưng tại vài iteration sẽ gọi VQLS
#     để so sánh direction. Đây là mode nên chạy đầu tiên.
#
# vqls:
#     dùng VQLS để update thật. Chỉ bật sau khi compare cho residual tốt.
#

PDIPM_LINEAR_SOLVER = "compare"   # "classical", "compare", or "vqls"

# Chỉ test VQLS tại các iteration này trong compare mode.
# Không nên test mọi iteration vì rất chậm.
VQLS_COMPARE_ITERS = (0, 1, 2, 3)

# Nếu dùng mode vqls, bắt đầu dùng VQLS từ iteration này.
PDIPM_VQLS_START_ITER = 0

# Nếu VQLS residual quá lớn khi đang dùng mode vqls, dừng để tránh update sai.
PDIPM_STOP_IF_VQLS_BAD = True
PDIPM_VQLS_MAX_REL_RESIDUAL = 1e-2


# ============================================================
# VQLS configuration
# ============================================================

VQLS_SEED = 123
VQLS_LAYERS = 4
VQLS_MAXITER = 800
VQLS_RESTARTS = 1
VQLS_INIT_SCALE = 0.10
VQLS_OPT_METHOD = "COBYLA"

VQLS_SCALE_SYSTEM = True
VQLS_USE_RUIZ_SCALING = True
VQLS_RUIZ_ITERS = 10

# SC-PDIPM case hiện tại có KKT 36x36, pad lên 64.
VQLS_MAX_PAD_DIM = 64

# ============================================================
# Output / plots
# ============================================================

OUTPUT_DIR = "outputs_qopf3"
SAVE_FIGURES = True


# ============================================================
# Which runs to execute
# ============================================================

RUN_CLASSICAL_BASELINE = True
RUN_VQLS_BASELINE = True


# ============================================================
# VQLS safety inside SC-PDIPM
# ============================================================

PDIPM_VQLS_MAX_REL_RESIDUAL = 1e-2
PDIPM_STOP_IF_VQLS_BAD = True

# ============================================================
# VQLS configuration
# ============================================================

VQLS_SEED = 123

# Tăng layers vì KKT 64 chiều không đơn giản
VQLS_LAYERS = 10

# 800 quá ít, COBYLA đang dừng vì MAXFUN
VQLS_MAXITER = 4000

# Chạy nhiều restart để tránh local minimum xấu
VQLS_RESTARTS = 3

# Scale khởi tạo
VQLS_INIT_SCALE = 0.20

VQLS_OPT_METHOD = "COBYLA"

VQLS_SCALE_SYSTEM = True
VQLS_USE_RUIZ_SCALING = True
VQLS_RUIZ_ITERS = 10

VQLS_MAX_PAD_DIM = 64

# Quan trọng: dùng real ansatz cho hệ KKT thực
VQLS_REAL_ANSATZ = True


# ============================================================
# VQLS safety
# ============================================================

PDIPM_STOP_IF_VQLS_BAD = True

# Giữ nghiêm ngặt. Nếu residual lớn hơn mức này thì không update.
PDIPM_VQLS_MAX_REL_RESIDUAL = 1e-2