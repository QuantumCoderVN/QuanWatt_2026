import numpy as np
import pandapower.networks as pn

np.set_printoptions(precision=6, suppress=True, linewidth=200)

# =========================
# Load IEEE 14-bus real data
# =========================
net = pn.case14()
Sbase = float(net.sn_mva)
nb = len(net.bus)

print("IEEE 14-bus real data from pandapower")
print("Base MVA:", Sbase)
print("Number of buses:", nb)

# =========================
# Generator buses
# ext_grid + gen
# =========================
gen_buses = []

for _, row in net.ext_grid.iterrows():
    gen_buses.append(int(row.bus))

for _, row in net.gen.iterrows():
    gen_buses.append(int(row.bus))

ng = len(gen_buses)

print("Number of generators including slack:", ng)
print("Generator buses:", gen_buses)

# =========================
# Build DC Bbus matrix from real IEEE lines/transformers
# =========================
Bbus = np.zeros((nb, nb))

# Lines
for _, row in net.line.iterrows():
    i = int(row.from_bus)
    j = int(row.to_bus)

    vn_kv = float(net.bus.loc[i, "vn_kv"])
    zbase = vn_kv**2 / Sbase

    x_ohm = float(row.x_ohm_per_km) * float(row.length_km)
    x_pu = x_ohm / zbase

    if abs(x_pu) < 1e-12:
        continue

    b = 1.0 / x_pu

    Bbus[i, i] += b
    Bbus[j, j] += b
    Bbus[i, j] -= b
    Bbus[j, i] -= b

# Transformers
for _, row in net.trafo.iterrows():
    i = int(row.hv_bus)
    j = int(row.lv_bus)

    x_pu = float(row.vk_percent) / 100.0 * Sbase / float(row.sn_mva)

    if abs(x_pu) < 1e-12:
        continue

    b = 1.0 / x_pu

    Bbus[i, i] += b
    Bbus[j, j] += b
    Bbus[i, j] -= b
    Bbus[j, i] -= b

print("\nBbus matrix:")
print(Bbus)

# =========================
# Decision variables:
# x = [theta_0 ... theta_13, Pg_0 ... Pg_4]
# =========================
nvar = nb + ng

# Real MATPOWER IEEE14 quadratic cost coefficients:
# gencost c2 for 5 generators
c2 = np.array([0.0430293, 0.25, 0.01, 0.01, 0.01])

H = np.zeros((nvar, nvar))
H[nb:, nb:] = np.diag(2.0 * c2)

print("\nHessian H:")
print(H)

# =========================
# Equality constraints:
# Bbus * theta - Cg * Pg = -Pd
# plus slack angle theta_slack = 0
# =========================
Cg = np.zeros((nb, ng))

for k, bus_id in enumerate(gen_buses):
    Cg[bus_id, k] = 1.0

A_power_balance = np.hstack([Bbus, -Cg])

slack_bus = int(net.ext_grid.iloc[0].bus)

A_slack = np.zeros((1, nvar))
A_slack[0, slack_bus] = 1.0

A = np.vstack([A_power_balance, A_slack])

print("\nJacobian / constraint matrix A:")
print(A)

# =========================
# KKT / Newton matrix
# This is the matrix OPF solvers solve
# =========================
KKT = np.block([
    [H, A.T],
    [A, np.zeros((A.shape[0], A.shape[0]))]
])

print("\nKKT matrix:")
print(KKT)

# =========================
# Eigenvalue check
# =========================
eig_H = np.linalg.eigvalsh(H)
eig_KKT = np.linalg.eigvalsh(KKT)

tol = 1e-9

print("\nEigenvalues of H:")
print(eig_H)

print("\nH positive definite?", np.all(eig_H > tol))
print("H positive semidefinite?", np.all(eig_H >= -tol))

print("\nEigenvalues of KKT:")
print(eig_KKT)

print("\nKKT positive definite?", np.all(eig_KKT > tol))
print("KKT positive semidefinite?", np.all(eig_KKT >= -tol))
print("KKT has negative eigenvalues?", np.any(eig_KKT < -tol))
print("KKT is indefinite?", np.any(eig_KKT > tol) and np.any(eig_KKT < -tol))